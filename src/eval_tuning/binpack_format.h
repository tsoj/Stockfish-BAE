/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2021 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <cstdio>
#include <cassert>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <fstream>
#include <cstring>
#include <iostream>
#include <set>
#include <cstdio>
#include <cassert>
#include <array>
#include <limits>
#include <climits>
#include <optional>
#include <iomanip>
#include <bit>

#include "../position.h"
#include "../movegen.h"
#include "../types.h"
#include "../uci.h"

#include "dataloader.h"

namespace util {

inline std::size_t usedBits(std::size_t value) {
    if (value == 0)
        return 0;
    return (63 ^ std::countl_zero(value)) + 1;
}

namespace lookup {
constexpr int nthSetBitIndexNaive(uint64_t value, int n) {
    for (int i = 0; i < n; ++i)
    {
        value &= value - 1;
    }
    return std::countr_zero(value);
}

constexpr std::array<std::array<uint8_t, 8>, 256> nthSetBitIndex = []() {
    std::array<std::array<uint8_t, 8>, 256> t{};

    for (int i = 0; i < 256; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            t[i][j] = nthSetBitIndexNaive(i, j);
        }
    }

    return t;
}();
}  // namespace lookup

inline int nthSetBitIndex(uint64_t v, uint64_t n) {

    std::uint64_t shift = 0;

    std::uint64_t p     = std::popcount(v & 0xFFFFFFFFull);
    std::uint64_t pmask = static_cast<std::uint64_t>(p > n) - 1ull;
    v >>= 32 & pmask;
    shift += 32 & pmask;
    n -= p & pmask;

    p     = std::popcount(v & 0xFFFFull);
    pmask = static_cast<std::uint64_t>(p > n) - 1ull;
    v >>= 16 & pmask;
    shift += 16 & pmask;
    n -= p & pmask;

    p     = std::popcount(v & 0xFFull);
    pmask = static_cast<std::uint64_t>(p > n) - 1ull;
    shift += 8 & pmask;
    v >>= 8 & pmask;
    n -= p & pmask;

    return static_cast<int>(lookup::nthSetBitIndex[v & 0xFFull][n] + shift);
}
}  // namespace util

namespace binpack {
constexpr std::size_t KiB = 1024;
constexpr std::size_t MiB = (1024 * KiB);
constexpr std::size_t GiB = (1024 * MiB);

constexpr std::size_t suggestedChunkSize = MiB;
constexpr std::size_t maxMovelistSize    = 10 * KiB;  // a safe upper bound
constexpr std::size_t maxChunkSize =
  100 * MiB;  // to prevent malformed files from causing huge allocations

using namespace std::literals;


struct XCompressedPosition {
    friend struct Position;

    // Occupied bitboard has bits set for
    // each square with a piece on it.
    // Each packedState byte holds 2 values (nibbles).
    // First one at low bits, second one at high bits.
    // Values correspond to consecutive squares
    // in bitboard iteration order.
    // Nibble values:
    // these are the same as for Piece
    // knights, bishops, queens can just be copied
    //  0 : white pawn
    //  1 : black pawn
    //  2 : white knight
    //  3 : black knight
    //  4 : white bishop
    //  5 : black bishop
    //  6 : white rook
    //  7 : black rook
    //  8 : white queen
    //  9 : black queen
    // 10 : white king
    // 11 : black king
    //
    // these are special
    // 12 : pawn with ep square behind (white or black, depending on rank)
    // 13 : white rook with coresponding castling rights
    // 14 : black rook with coresponding castling rights
    // 15 : black king and black is side to move
    //
    // Let N be the number of bits set in occupied bitboard.
    // Only N nibbles are present. (N+1)/2 bytes are initialized.

    static XCompressedPosition readFromBigEndian(const unsigned char* data) {
        XCompressedPosition pos{};
        pos.m_occupied =
          Bitboard((uint64_t) data[0] << 56 | (uint64_t) data[1] << 48 | (uint64_t) data[2] << 40
                   | (uint64_t) data[3] << 32 | (uint64_t) data[4] << 24 | (uint64_t) data[5] << 16
                   | (uint64_t) data[6] << 8 | (uint64_t) data[7]);
        std::memcpy(pos.m_packedState, data + 8, 16);
        return pos;
    }

    constexpr XCompressedPosition() :
        m_occupied{},
        m_packedState{} {}

    [[nodiscard]] friend bool operator<(const XCompressedPosition& lhs,
                                        const XCompressedPosition& rhs) {
        if (lhs.m_occupied < rhs.m_occupied)
            return true;
        if (lhs.m_occupied > rhs.m_occupied)
            return false;

        return std::strcmp(reinterpret_cast<const char*>(lhs.m_packedState),
                           reinterpret_cast<const char*>(rhs.m_packedState))
             < 0;
    }

    [[nodiscard]] friend bool operator==(const XCompressedPosition& lhs,
                                         const XCompressedPosition& rhs) {
        return lhs.m_occupied == rhs.m_occupied
            && std::strcmp(reinterpret_cast<const char*>(lhs.m_packedState),
                           reinterpret_cast<const char*>(rhs.m_packedState))
                 == 0;
    }

    [[nodiscard]] inline Position decompress() const;

    [[nodiscard]] constexpr Bitboard pieceBB() const { return m_occupied; }

    void writeToBigEndian(unsigned char* data) {
        const Bitboard occupied = m_occupied;
        *data++                 = occupied >> 56;
        *data++                 = (occupied >> 48) & 0xFF;
        *data++                 = (occupied >> 40) & 0xFF;
        *data++                 = (occupied >> 32) & 0xFF;
        *data++                 = (occupied >> 24) & 0xFF;
        *data++                 = (occupied >> 16) & 0xFF;
        *data++                 = (occupied >> 8) & 0xFF;
        *data++                 = occupied & 0xFF;
        std::memcpy(data, m_packedState, 16);
    }

   private:
    Bitboard m_occupied;
    uint8_t  m_packedState[16];
};

static_assert(sizeof(XCompressedPosition) == 24);
static_assert(std::is_trivially_copyable_v<XCompressedPosition>);

inline void set_castling_right(Position* pos, Color c, Square rfrom) {
    assert(pos != nullptr);
    Square kfrom = c == WHITE ? SQ_E1 : SQ_E8;
    ;
    CastlingRights cr = c & (kfrom < rfrom ? KING_SIDE : QUEEN_SIDE);

    pos->st->castlingRights |= cr;
    pos->castlingRightsMask[kfrom] |= cr;
    pos->castlingRightsMask[rfrom] |= cr;
    pos->castlingRookSquare[cr] = rfrom;

    Square kto = relative_square(c, cr & KING_SIDE ? SQ_G1 : SQ_C1);
    Square rto = relative_square(c, cr & KING_SIDE ? SQ_F1 : SQ_D1);

    pos->castlingPath[cr] =
      (between_bb(rfrom, rto) | between_bb(kfrom, kto) | rto | kto) & ~(kfrom | rfrom);
}

[[nodiscard]] inline Position XCompressedPosition::decompress() const {
    //std::cout << "decompress start" << std::endl;
    Position pos{};
    //std::cout << "ep square: " << UCI::square(pos.ep_square()) << std::endl;


    const auto decompressPiece = [&pos](const Square sq, const uint8_t nibble) {
        //std::cout << "square: " << UCI::square(sq) << ", nibble: " << static_cast<int>(nibble)
        //          << std::endl;
        switch (nibble)
        {
            // clang-format off
        case 0 : pos.put_piece(W_PAWN, sq); return;
        case 1 : pos.put_piece(B_PAWN, sq); return;
        case 2 : pos.put_piece(W_KNIGHT, sq); return;
        case 3 : pos.put_piece(B_KNIGHT, sq); return;
        case 4 : pos.put_piece(W_BISHOP, sq); return;
        case 5 : pos.put_piece(B_BISHOP, sq); return;
        case 6 : pos.put_piece(W_ROOK, sq); return;
        case 7 : pos.put_piece(B_ROOK, sq); return;
        case 8 : pos.put_piece(W_QUEEN, sq); return;
        case 9 : pos.put_piece(B_QUEEN, sq); return;
        case 10 : pos.put_piece(W_KING, sq); return;
        case 11 : pos.put_piece(B_KING, sq); return;
        // clang-format om
        case 12 : {
            const Rank rank = rank_of(sq);
            if (rank == RANK_4)
            {
                pos.put_piece(W_PAWN, sq);
                pos.st->epSquare = sq + SOUTH;
            }
            else
            {
                assert(rank == RANK_5);
                pos.put_piece(B_PAWN, sq);
                pos.st->epSquare = sq + NORTH;
            }
            return;
        }

        case 13 : {
            pos.put_piece(W_ROOK, sq);
            if (sq == SQ_A1)
            {
                set_castling_right(&pos, WHITE, SQ_A1);
            }
            else
            {
                assert(sq == SQ_H1);
                set_castling_right(&pos, WHITE, SQ_H1);
            }
            return;
        }

        case 14 : {
            pos.put_piece(B_ROOK, sq);
            if (sq == SQ_A8)
            {
                set_castling_right(&pos, BLACK, SQ_A8);
            }
            else
            {
                assert(sq == SQ_H8);
                set_castling_right(&pos, BLACK, SQ_H8);
            }
            return;
        }

        case 15 : {
            pos.put_piece(B_KING, sq);
            pos.sideToMove = BLACK;
            return;
        }
        }

        return;
    };

    Bitboard occ = m_occupied;

    for (int i = 0;; ++i)
    {
        if (occ == 0)
            break;
        decompressPiece(lsb(occ), m_packedState[i] & 0xF);
        occ &= occ - 1;

        if (occ == 0)
            break;
        decompressPiece(lsb(occ), m_packedState[i] >> 4);
        occ &= occ - 1;
    }
    //std::cout << "decompress finish" << std::endl;

    return pos;
}


struct XCompressedMove {
   private:
    // from most significant bits
    // 2 bits for move type
    // 6 bits for from square
    // 6 bits for to square
    // 2 bits for promoted piece type
    //    0 if not a promotion
    static constexpr uint16_t squareMask            = 0b111111u;
    static constexpr uint16_t promotedPieceTypeMask = 0b11u;
    static constexpr uint16_t moveTypeMask          = 0b11u;

   public:
    [[nodiscard]] constexpr static XCompressedMove readFromBigEndian(const unsigned char* data) {
        XCompressedMove move{};
        move.m_packed = (data[0] << 8) | data[1];
        return move;
    }

    constexpr XCompressedMove() noexcept :
        m_packed(0) {}

    [[nodiscard]] constexpr uint16_t packed() const { return m_packed; }

    [[nodiscard]] constexpr Move decompress(const Position& pos) const noexcept {
        if (m_packed == 0)
        {
            return MOVE_NONE;
        }
        else
        {
            const MoveType type = [&]() {
                switch (m_packed >> (16 - 2))
                {
                case 0 :
                    return NORMAL;
                case 1 :
                    return PROMOTION;
                case 2 :
                    return CASTLING;
                case 3 :
                    return ENPASSANT;
                default :
                    assert(false);
                    return NORMAL;
                }
            }();
            const Square from          = Square((m_packed >> (16 - 2 - 6)) & squareMask);
            const Square to            = Square((m_packed >> (16 - 2 - 6 - 6)) & squareMask);
            const Piece  promotedPiece = [&]() {
                if (type == PROMOTION)
                {
                    const Color color = (rank_of(to) == RANK_1) ? BLACK : WHITE;

                    const PieceType pt = PieceType((m_packed & promotedPieceTypeMask) + KNIGHT);
                    return make_piece(color, pt);
                }
                else
                {
                    return NO_PIECE;
                }
            }();



            //std::cout << "some pos: " << pos.fen() << std::endl;
            for (const auto& m : MoveList<LEGAL>(pos))
            {
                //std::cout << "legal move: " << UCI::move(m, false) << std::endl;
                if (from_sq(m) == from && to_sq(m) == to && type_of(m) == type
                    && (type != PROMOTION || promotion_type(m) == type_of(promotedPiece)))
                {
                    return m;
                }
            }

            assert(false);
            return MOVE_NONE;
        }
    }

   private:
    uint16_t m_packed;
};

static_assert(sizeof(XCompressedMove) == 2);

struct CompressedTrainingDataFile {
    struct Header {
        uint32_t chunkSize;
    };

    CompressedTrainingDataFile(std::string path, std::ios_base::openmode om = std::ios_base::app) :
        m_path(std::move(path)),
        m_file(m_path, std::ios_base::binary | std::ios_base::in | std::ios_base::out | om) {
        // Necessary for MAC because app mode makes it put the reading
        // head at the end.
        m_file.seekg(0);
    }

    void append(const char* data, uint32_t size) {
        writeChunkHeader({size});
        m_file.write(data, size);
    }

    [[nodiscard]] bool hasNextChunk() {
        if (!m_file)
        {
            return false;
        }

        m_file.peek();
        return !m_file.eof();
    }

    [[nodiscard]] std::vector<unsigned char> readNextChunk() {
        auto                       size = readChunkHeader().chunkSize;
        std::vector<unsigned char> data(size);
        m_file.read(reinterpret_cast<char*>(data.data()), size);
        return data;
    }

   private:
    std::string  m_path;
    std::fstream m_file;

    void writeChunkHeader(Header h) {
        unsigned char header[8];
        header[0] = 'B';
        header[1] = 'I';
        header[2] = 'N';
        header[3] = 'P';
        header[4] = h.chunkSize;
        header[5] = h.chunkSize >> 8;
        header[6] = h.chunkSize >> 16;
        header[7] = h.chunkSize >> 24;
        m_file.write(reinterpret_cast<const char*>(header), 8);
    }

    [[nodiscard]] Header readChunkHeader() {
        unsigned char header[8];
        m_file.read(reinterpret_cast<char*>(header), 8);
        if (header[0] != 'B' || header[1] != 'I' || header[2] != 'N' || header[3] != 'P')
        {
            assert(false);
            // throw std::runtime_error("Invalid binpack file or chunk.");
        }

        const uint32_t size = header[4] | (header[5] << 8) | (header[6] << 16) | (header[7] << 24);

        if (size > maxChunkSize)
        {
            assert(false);
            // throw std::runtime_error("Chunks size larger than supported. Malformed file?");
        }

        return {size};
    }
};

[[nodiscard]] inline uint16_t signedToUnsigned(int16_t a) {
    uint16_t r;
    std::memcpy(&r, &a, sizeof(uint16_t));
    if (r & 0x8000)
    {
        r ^= 0x7FFF;
    }
    r = (r << 1) | (r >> 15);
    return r;
}

[[nodiscard]] inline int16_t unsignedToSigned(uint16_t r) {
    int16_t a;
    r = (r << 15) | (r >> 1);
    if (r & 0x8000)
    {
        r ^= 0x7FFF;
    }
    std::memcpy(&a, &r, sizeof(uint16_t));
    return a;
}

struct TrainingDataEntry {
    Position pos;
    Move     move;
    int16_t  score;
    uint16_t ply;
    int16_t  result;

    [[nodiscard]] bool isValid() const { return pos.legal(move); }

    [[nodiscard]] bool isCapturingMove() const { return pos.capture(move); }

    [[nodiscard]] bool isInCheck() const { return pos.checkers() != 0; }
};

struct PackedTrainingDataEntry {
    unsigned char bytes[32];
};

[[nodiscard]] inline std::size_t usedBitsSafe(std::size_t value) {
    if (value == 0)
        return 0;
    return util::usedBits(value - 1);
}

static constexpr std::size_t scoreVleBlockSize = 4;

int aoisjdoasi = 0;
struct PackedMoveScoreListReader {
    TrainingDataEntry entry;
    uint16_t          numPlies;
    unsigned char*    movetext;

    PackedMoveScoreListReader(const TrainingDataEntry& entry_,
                              unsigned char*           movetext_,
                              uint16_t                 numPlies_) :
        entry(entry_),
        numPlies(numPlies_),
        movetext(movetext_),
        m_lastScore(-entry_.score) {}

    [[nodiscard]] uint8_t extractBitsLE8(std::size_t count) {
        if (count == 0)
            return 0;

        if (m_readBitsLeft == 0)
        {
            m_readOffset += 1;
            m_readBitsLeft = 8;
        }

        const uint8_t byte = movetext[m_readOffset] << (8 - m_readBitsLeft);
        uint8_t       bits = byte >> (8 - count);

        if (count > m_readBitsLeft)
        {
            const auto spillCount = count - m_readBitsLeft;
            bits |= movetext[m_readOffset + 1] >> (8 - spillCount);

            m_readBitsLeft += 8;
            m_readOffset += 1;
        }

        m_readBitsLeft -= count;

        return bits;
    }

    [[nodiscard]] uint16_t extractVle16(std::size_t blockSize) {
        auto        mask   = (1 << blockSize) - 1;
        uint16_t    v      = 0;
        std::size_t offset = 0;
        for (;;)
        {
            uint16_t block = extractBitsLE8(blockSize + 1);
            v |= ((block & mask) << offset);
            if (!(block >> blockSize))
            {
                break;
            }

            offset += blockSize;
        }
        return v;
    }


    [[nodiscard]] TrainingDataEntry nextEntry() {
        aoisjdoasi  += 1;
        auto newSt = std::make_unique<StateInfo>();
        Color us = entry.pos.side_to_move();
        Piece  captured = type_of(entry.move) == ENPASSANT ? make_piece(~us, PAWN) : entry.pos.piece_on(to_sq(entry.move));
        if(!(captured == NO_PIECE || color_of(captured) == (type_of(entry.move) != CASTLING ? ~us : us))){
            std::cout << aoisjdoasi << std::endl;
            std::cout << entry.pos.fen() << std::endl;
            std::cout << UCI::move(entry.move, false) << ", to: " << to_sq(entry.move) << std::endl;
            std::cout << captured << std::endl;
            std::cout << type_of(entry.move) << std::endl;
        }
        assert(captured == NO_PIECE || color_of(captured) == (type_of(entry.move) != CASTLING ? ~us : us));
        //std::cout << "pos before: " << entry.pos.fen() << std::endl;
        entry.pos.do_move(entry.move, *newSt);
        //std::cout << "pos after: " << entry.pos.fen() << std::endl;
        entry.pos.own_st = std::move(newSt);

        auto [move, score] = nextMoveScore(entry.pos);
        assert(entry.pos.legal(move));
        //std::cout << "next move: " << UCI::move(move, false) << ", to: " << to_sq(move) << std::endl;
        entry.move         = move;
        entry.score        = score;
        entry.ply += 1;
        entry.result = -entry.result;
        return entry;
    }

    [[nodiscard]] bool hasNext() const { return m_numReadPlies < numPlies; }

    [[nodiscard]] std::pair<Move, int16_t> nextMoveScore(const Position& pos) {
        Move    move;
        int16_t score;

        const bool print = false;//aoisjdoasi == 20428910;//20428914;

        const Color    sideToMove  = pos.side_to_move();
        const Bitboard ourPieces   = pos.pieces(sideToMove);
        const Bitboard theirPieces = pos.pieces(~sideToMove);
        const Bitboard occupied    = ourPieces | theirPieces;
        if(print) std::cout << "pos: " << pos.fen() << std::endl;
        if(print) std::cout << "sideToMove: " << sideToMove << std::endl;

        const auto   pieceId = extractBitsLE8(usedBitsSafe(popcount(ourPieces)));
        if(print) std::cout << "ourPieces: " << static_cast<uint64_t>(ourPieces) << std::endl;
        if(print) std::cout << "ourPieces.count(): " << popcount(ourPieces) << std::endl;
        if(print) std::cout << "usedBitsSafe(popcount(ourPieces)): " << usedBitsSafe(popcount(ourPieces)) << std::endl;
        if(print) std::cout << "pieceId: " << static_cast<int>(pieceId) << std::endl;
        const Square from    = Square(util::nthSetBitIndex(ourPieces, pieceId));
        if(print) std::cout << "from: " << from << std::endl;

        const PieceType pt = type_of(pos.piece_on(from));
        switch (pt)
        {
        case PAWN : {
            const Rank      promotionRank = pos.side_to_move() == WHITE ? RANK_7 : RANK_2;
            const Rank      startRank     = pos.side_to_move() == WHITE ? RANK_2 : RANK_7;
            const Direction forward       = sideToMove == WHITE ? NORTH : SOUTH;

            if(print) std::cout << "PAWN" << std::endl;
            if(print) std::cout << pos.fen() << std::endl;

            const Square epSquare = pos.ep_square();

            Bitboard attackTargets = theirPieces;
            if(print) std::cout << "attackTargets A: " << attackTargets << std::endl;
            if (epSquare != SQ_NONE)
            {
                for (const auto& m : MoveList<LEGAL>(pos))
                {
                    if(type_of(m) == ENPASSANT && to_sq(m) == epSquare)
                    {
                        if(print) std::cout << "PAWN A: " << epSquare << " (" << UCI::square(epSquare) << ")" << std::endl;
                        attackTargets |= square_bb(epSquare);
                        break;

                    }
                }
            }

            Bitboard destinations = pawn_attacks_bb(sideToMove, from) & attackTargets;
            if(print) std::cout << "pawn_attacks_bb(sideToMove, from): " << pawn_attacks_bb(sideToMove, from) << std::endl;
            if(print) std::cout << "attackTargets: " << attackTargets << std::endl;
            if(print) std::cout << "destinations A: " << destinations << std::endl;

            const Square sqForward = from + forward;
            if(print) std::cout << "sqForward: "<< UCI::square(sqForward) << std::endl;
            if ((occupied & square_bb(sqForward)) == 0)
            {
                if(print) std::cout << "PAWN B" << std::endl;
                destinations |= square_bb(sqForward);
                if (rank_of(from) == startRank && (occupied & square_bb(sqForward + forward)) == 0)
                {
                    destinations |= square_bb(sqForward + forward);
                }
            }
            if(print) std::cout << "destinations B: " << destinations << std::endl;

            const auto destinationsCount = popcount(destinations);
            if(print) std::cout << "destinationsCount: "<< destinationsCount << std::endl;
            if (rank_of(from) == promotionRank)
            {
                const auto      moveId = extractBitsLE8(usedBitsSafe(destinationsCount * 4ull));
                const PieceType promotedPiece = static_cast<PieceType>(KNIGHT + (moveId % 4ull));
                const Square    to = Square(util::nthSetBitIndex(destinations, moveId / 4ull));

                // move = chess::Move::promotion(from, to, promotedPiece);
                if(print) std::cout << "PAWN C" << std::endl;
                move = make<PROMOTION>(from, to, promotedPiece);
                break;
            }
            else
            {
                const auto   moveId = extractBitsLE8(usedBitsSafe(destinationsCount));
                const Square to     = Square(util::nthSetBitIndex(destinations, moveId));

                        if(print) std::cout << "destinationsCount: " << destinationsCount << std::endl;
                        if(print) std::cout << "usedBitsSafe(destinationsCount): " << usedBitsSafe(destinationsCount) << std::endl;
                        if(print) std::cout << "moveId: " << moveId << std::endl;
                        if(print) std::cout << "destinations.bits(): " << destinations << std::endl;
                        if(print) std::cout << "chess::nthSetBitIndex(destinations.bits(), moveId): " << util::nthSetBitIndex(destinations, moveId) << std::endl;

                if (to == epSquare)
                {
                    if(print) std::cout << "PAWN D" << std::endl;
                    move = make<ENPASSANT>(from, to);
                    break;
                }
                else
                {
                    if(print) std::cout << "PAWN E, from: " << from << " (" << UCI::square(from) << "), to: " << to << std::endl;
                    move = make_move(from, to);
                    break;
                }
            }
        }
        case KING : {
            const CastlingRights ourCastlingRightsMask = pos.castling_rights(sideToMove);

            const Bitboard    attacks     = attacks_bb<KING>(from, 0) & ~ourPieces;
            const std::size_t attacksSize = popcount(attacks);
            const std::size_t numCastlings =
              std::popcount(static_cast<uint64_t>(ourCastlingRightsMask));

            const auto moveId = extractBitsLE8(usedBitsSafe(attacksSize + numCastlings));

            if (moveId >= attacksSize)
            {
                const std::size_t idx = moveId - attacksSize;

                CastlingRights castleType =
                  idx == 0 && (ourCastlingRightsMask & QUEEN_SIDE) != 0 ? QUEEN_SIDE : KING_SIDE;

                castleType = static_cast<CastlingRights>(
                  castleType & (sideToMove == WHITE ? WHITE_CASTLING : BLACK_CASTLING));

                move = make<CASTLING>(pos.square<KING>(sideToMove),
                                      pos.castling_rook_square(castleType));
                break;
            }
            else
            {
                const Square to = Square(util::nthSetBitIndex(attacks, moveId));
                move            = make_move(from, to);
                break;
            }
            break;
        }
        default : {
            const Bitboard attacks = attacks_bb(pt, from, occupied) & ~ourPieces;
            const auto     moveId  = extractBitsLE8(usedBitsSafe(popcount(attacks)));
            Square         to      = Square(util::nthSetBitIndex(attacks, moveId));
            //std::cout << "A: from: " << from << ", to: " << to << ", piece type: " << pt << std::endl;
            move                   = make_move(from, to);
            break;
        }
        }

        score       = m_lastScore + unsignedToSigned(extractVle16(scoreVleBlockSize));
        m_lastScore = -score;

        ++m_numReadPlies;

        if(print )
        {
            std::cout << from_sq(move) << ", " << to_sq(move) << std::endl;
            std::cout << UCI::move(move, false) << std::endl;
            exit(1);
        }

        return {move, score};
    }

    [[nodiscard]] std::size_t numReadBytes() { return m_readOffset + (m_readBitsLeft != 8); }

   private:
    std::size_t m_readBitsLeft = 8;
    std::size_t m_readOffset   = 0;
    int16_t     m_lastScore    = 0;
    uint16_t    m_numReadPlies = 0;
};


[[nodiscard]] inline TrainingDataEntry unpackEntry(const PackedTrainingDataEntry& packed) {
    TrainingDataEntry plain;

    std::size_t offset        = 0;
    auto        compressedPos = XCompressedPosition::readFromBigEndian(packed.bytes);
    plain.pos                 = compressedPos.decompress();
    //std::cout << plain.pos.fen() << std::endl;
    offset += sizeof(compressedPos);
    auto compressedMove = XCompressedMove::readFromBigEndian(packed.bytes + offset);
    plain.move          = compressedMove.decompress(plain.pos);
    offset += sizeof(compressedMove);
    plain.score = unsignedToSigned((packed.bytes[offset] << 8) | packed.bytes[offset + 1]);
    offset += 2;
    uint16_t pr       = (packed.bytes[offset] << 8) | packed.bytes[offset + 1];
    plain.ply         = pr & 0x3FFF;
    plain.pos.gamePly = plain.ply;
    plain.result      = unsignedToSigned(pr >> 14);
    offset += 2;
    plain.pos.st->rule50 = (packed.bytes[offset] << 8) | packed.bytes[offset + 1];

    return plain;
}

struct CompressedTrainingDataEntryReader {
    static constexpr std::size_t chunkSize = suggestedChunkSize;

    CompressedTrainingDataEntryReader(std::string             path,
                                      std::ios_base::openmode om = std::ios_base::app) :
        m_inputFile(path, om),
        m_chunk(),
        m_movelistReader(std::nullopt),
        m_offset(0),
        m_isEnd(false) {
        if (!m_inputFile.hasNextChunk())
        {
            std::cerr << "\"" << path << "\" contains nothing" << std::endl;
            m_isEnd = true;
        }
        else
        {
            m_chunk = m_inputFile.readNextChunk();
        }
    }

    [[nodiscard]] bool hasNext() { return !m_isEnd; }

    [[nodiscard]] TrainingDataEntry next() {
        if (m_movelistReader.has_value())
        {
            const auto e = m_movelistReader->nextEntry();

            if (!m_movelistReader->hasNext())
            {
                m_offset += m_movelistReader->numReadBytes();
                m_movelistReader.reset();

                fetchNextChunkIfNeeded();
            }

            return e;
        }

        PackedTrainingDataEntry packed;
        std::memcpy(&packed, m_chunk.data() + m_offset, sizeof(PackedTrainingDataEntry));
        m_offset += sizeof(PackedTrainingDataEntry);

        const uint16_t numPlies = (m_chunk[m_offset] << 8) | m_chunk[m_offset + 1];
        m_offset += 2;

        const auto e = unpackEntry(packed);

        if (numPlies > 0)
        {
            m_movelistReader.emplace(e, reinterpret_cast<unsigned char*>(m_chunk.data()) + m_offset,
                                     numPlies);
        }
        else
        {
            fetchNextChunkIfNeeded();
        }

        return e;
    }

   private:
    CompressedTrainingDataFile               m_inputFile;
    std::vector<unsigned char>               m_chunk;
    std::optional<PackedMoveScoreListReader> m_movelistReader;
    std::size_t                              m_offset;
    bool                                     m_isEnd;

    void fetchNextChunkIfNeeded() {
        if (m_offset + sizeof(PackedTrainingDataEntry) + 2 > m_chunk.size())
        {
            if (m_inputFile.hasNextChunk())
            {
                m_chunk  = m_inputFile.readNextChunk();
                m_offset = 0;
            }
            else
            {
                m_isEnd = true;
            }
        }
    }
};

inline float convert_result(float result) {
    if (result == 0)
        return 0.5;
    else if (result == -1)
        return 0;
    return 1;
}

inline float invert_wdl(float result) {
    if (result == 1)
        return 0;
    else if (result == 0)
        return 1;
    return result;
}

// Need to declare our own bb manip because sf sucks
inline int GetLsbIndex(uint64_t bitboard) { return std::countr_zero(bitboard); }


constexpr int bulletformatpiece[12] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13};

inline void emitPlainEntry(std::string& buffer, const TrainingDataEntry& plain) {
    // extract the result and score
    auto score = plain.score;

    auto result = convert_result(plain.result);
    // convert the result and score to white pov
    if (plain.pos.side_to_move() == BLACK)
    {
        score *= -1;
        result = invert_wdl(result);
    }

    // delay the data dumping to make sure you don't write stuff you are supposed to skip to the buffer
    buffer += plain.pos.fen();
    buffer += " | ";

    buffer += std::to_string(score);
    buffer += " | ";

    // Absolutely terrible hack because i can't get std::format to work on gcc 11
    std::ostringstream str{};
    str << std::setprecision(1) << result;
    buffer += str.str();
    buffer += "\n";
}
inline void validateBinpack(const std::string& inputPath) {
    constexpr std::size_t reportSize = 1000000;

    std::cout << "Validating " << inputPath << '\n';

    CompressedTrainingDataEntryReader reader(inputPath);
    std::size_t                       numProcessedPositions      = 0;
    std::size_t                       numProcessedPositionsBatch = 0;

    while (reader.hasNext())
    {
        auto e = reader.next();
        if (!e.isValid())
        {
            std::cerr << "Illegal move " << UCI::move(e.move, false) << " for position "
                      << e.pos.fen() << '\n';
            return;
        }

        ++numProcessedPositions;
        ++numProcessedPositionsBatch;

        if (numProcessedPositionsBatch >= reportSize)
        {
            numProcessedPositionsBatch -= reportSize;
            std::cout << "Processed " << numProcessedPositions << " positions.\n";
        }
    }

    if (numProcessedPositionsBatch)
    {
        std::cout << "Processed " << numProcessedPositions << " positions.\n";
    }

    std::cout << "Finished. Validated " << numProcessedPositions << " positions.\n";
}
} // namespace binpack



template<float wdlRatio>
class BinpackReader {
   public:
    explicit BinpackReader(const std::filesystem::path& path) : m_reader(path.string()) {
        std::cout << "Opened " << path << std::endl;
    }

    [[nodiscard]] std::optional<BufferEntry> next() {
        while (m_reader.hasNext())
        {
            const auto e = m_reader.next();
            if (e.isInCheck() || e.isCapturingMove() || std::abs(e.score) > 10'000)
            {
                continue;
            }

            const float probability = Eval::winningProbability(static_cast<Value>(e.score) * (e.pos.side_to_move() == BLACK ? -1 : 1));
            float outcome = binpack::convert_result(e.result);
            if(e.pos.side_to_move() == BLACK)
            {
                outcome = binpack::invert_wdl(outcome);
            }
            static_assert(wdlRatio >= 0.0F);
            static_assert(wdlRatio <= 1.0F);
            outcome = outcome * wdlRatio + probability * (1.0F - wdlRatio);
            assert(outcome <= 1.0);
            assert(outcome >= 0.0);
            return BufferEntry{Eval::toEvalPosition(e.pos), outcome};
        }
        return std::nullopt;
    }

   private:
    binpack::CompressedTrainingDataEntryReader m_reader;
};

template<float wdlRatio>
using BinpackDataloader = Dataloader<BinpackReader<wdlRatio>>;
