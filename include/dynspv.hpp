// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <spirv/unified1/spirv.hpp>

namespace dynspv
{
	template<typename T>
	concept spvConstant = (std::integral<T> || std::floating_point<T>);

	using IdMemorySemantics = spv::Id;
	using IdRef = spv::Id;
	using IdResult = spv::Id;
	using IdResultType = spv::Id;
	using IdScope = spv::Id;

	inline void countOperandsWord(uint16_t& wordCount)
	{
	}

	inline void countOperandWord(uint16_t& wordCount, spvConstant auto operand)
	{
		wordCount += ((sizeof(decltype(operand)) + 3) & ~0x3) / sizeof(uint32_t);
	}

	inline void countOperandWord(uint16_t& wordCount, const std::string& operand)
	{
		wordCount += operand.size() / sizeof(uint32_t) + 1;
	}

	inline void countOperandWord(uint16_t& wordCount, uint32_t operand)
	{
		wordCount++;
	}

	template<typename... TArgs>
	inline void countOperandWord(uint16_t& wordCount, const std::tuple<TArgs...>& operand)
	{
		std::apply([&wordCount](auto&... args) { (countOperandWord(wordCount, args), ...); }, operand);
	}

	template<typename T>
	inline void countOperandWord(uint16_t& wordCount, std::optional<T>& operand)
	{
		if (operand.has_value())
		{
			countOperandWord(wordCount, operand.value());
		}
	}

	template<typename T>
	inline void countOperandWord(uint16_t& wordCount, const std::vector<T>& operand)
	{
		for (auto&& el : operand)
		{
			countOperandWord(wordCount, el);
		}
	}

	template<typename T, typename... TArgs>
	inline void countOperandsWord(uint16_t& wordCount, T operand, TArgs... args)
	{
		countOperandWord(wordCount, operand);
		countOperandsWord(wordCount, std::forward<TArgs>(args)...);
	}

	constexpr size_t DEFAULT_MAX_CODE_SIZE = 1024;
	constexpr size_t BOUND_INDEX = 3;

	class ModuleGenerator
	{
	  protected:
		std::vector<uint32_t> m_code{DEFAULT_MAX_CODE_SIZE};
		size_t m_size{0};

		void growMemory()
		{
			m_code.resize(std::max(DEFAULT_MAX_CODE_SIZE, m_code.size() * 2));
		}

		uint32_t m_id = 1;

		uint32_t nextId()
		{
			return m_id++;
		}

	  public:
		uint32_t getBound() const
		{
			return m_id;
		}

		const std::vector<uint32_t>& getCode()
		{
			if (m_code.size() != m_size)
			{
				m_code.resize(m_size);
			}

			return m_code;
		}

		inline void writeWord(uint32_t val)
		{
			if (m_size >= m_code.size())
			{
				growMemory();
			}

			m_code[m_size++] = val;
		}

		inline void writeWord(uint16_t low, uint16_t high)
		{
			uint32_t word = (static_cast<uint32_t>(high) << 16) | static_cast<uint32_t>(low);
			writeWord(word);
		}

		inline void writeWord(spvConstant auto val)
		{
			using T = decltype(val);
			constexpr size_t words_count = ((sizeof(T) + 3) & ~0x3) / sizeof(uint32_t);
			union
			{
				T v;
				uint32_t u32[words_count];
			} vals = {val};

			if constexpr (!std::is_floating_point_v<T> && std::is_signed_v<T> && sizeof(T) < sizeof(uint32_t))
			{
				vals.u32[0] = (uint32_t)(int32_t)(T)vals.u32[0];
			}

			for (size_t i = 0; i < words_count; i++)
			{
				writeWord(vals.u32[i]);
			}
		}

		inline void writeWord(const std::string& string)
		{
			const uint32_t* vals = reinterpret_cast<const uint32_t*>(string.data());
			const size_t size = string.size() / sizeof(uint32_t);
			size_t bytesWritten = 0;

			for (size_t i = 0; i < size; i++)
			{
				writeWord(vals[i]);
				bytesWritten += 4;
			}

			if (bytesWritten == string.size())
			{
				writeWord(0);
				return;
			}

			uint32_t lastVal = 0;
			for (size_t i = bytesWritten; i < string.size(); i++)
			{
				lastVal = lastVal << 8 | string[i];
			}

			writeWord(lastVal);
		}

		template<typename... TArgs>
		inline void writeWord(const std::tuple<TArgs...>& val)
		{
			std::apply([this](auto&... args) { (writeWord(args), ...); }, val);
		}

		template<typename T>
			requires std::is_enum_v<T>
		inline void writeWord(T val)
		{
			writeWord(static_cast<uint32_t>(val));
		}

		template<typename T>
		inline void writeWord(const std::vector<T>& values)
		{
			for (auto&& val : values)
			{
				writeWord(val);
			}
		}

		template<typename T>
		inline void writeWord(std::optional<T> word)
		{
			if (word.has_value())
			{
				writeWord(word.value());
			}
		}

		inline void writeWords()
		{
		}

		template<typename T, typename... TArgs>
		inline void writeWords(T val, TArgs... args)
		{
			writeWord(val);
			writeWords(std::forward<TArgs>(args)...);
		}

		void writeMagicNumber()
		{
			writeWord(spv::MagicNumber);
		}

		void writeVersionNumber(uint32_t version = spv::Version)
		{
			writeWord(version);
		}

		void writeGeneratorMagicNumber()
		{
			writeWord(0);
		}

		void writeBound(uint32_t bound = 0)
		{
			writeWord(bound);
		}

		void updateBound(uint32_t bound)
		{
			size_t oldSize = m_size;
			m_size = BOUND_INDEX;
			writeWord(bound);
			m_size = oldSize;
		}

		void writeInstructionSchema()
		{
			writeWord(0);
		}

		void writeHeader(uint32_t version = spv::Version)
		{
			writeMagicNumber();
			writeVersionNumber(version);
			writeGeneratorMagicNumber();
			writeBound();
			writeInstructionSchema();
		}

		void OpAbsISubINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpAbsISubINTEL, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpAbsUSubINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpAbsUSubINTEL, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpAccessChain(
			IdResultType idResultType,
			IdResult idResult,
			IdRef base,
			const std::vector<IdRef>& indexes = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, indexes);

			writeWord(spv::Op::OpAccessChain, wordCount);
			writeWords(idResultType, idResult, base, indexes);
		}

		void OpAliasDomainDeclINTEL(
			IdResult idResult,
			std::optional<IdRef> name = {})
		{
			uint16_t wordCount = 2;
			countOperandsWord(wordCount, name);

			writeWord(spv::Op::OpAliasDomainDeclINTEL, wordCount);
			writeWords(idResult, name);
		}

		void OpAliasScopeDeclINTEL(
			IdResult idResult,
			IdRef aliasDomain,
			std::optional<IdRef> name = {})
		{
			uint16_t wordCount = 3;
			countOperandsWord(wordCount, name);

			writeWord(spv::Op::OpAliasScopeDeclINTEL, wordCount);
			writeWords(idResult, aliasDomain, name);
		}

		void OpAliasScopeListDeclINTEL(
			IdResult idResult,
			const std::vector<IdRef>& aliasScopes = {})
		{
			uint16_t wordCount = 2;
			countOperandsWord(wordCount, aliasScopes);

			writeWord(spv::Op::OpAliasScopeListDeclINTEL, wordCount);
			writeWords(idResult, aliasScopes);
		}

		void OpAll(
			IdResultType idResultType,
			IdResult idResult,
			IdRef vector)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpAll, wordCount);
			writeWords(idResultType, idResult, vector);
		}

		void OpAllocateNodePayloadsAMDX(
			IdResultType idResultType,
			IdResult idResult,
			IdScope visibility,
			IdRef payloadCount,
			IdRef nodeIndex)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpAllocateNodePayloadsAMDX, wordCount);
			writeWords(idResultType, idResult, visibility, payloadCount, nodeIndex);
		}

		void OpAny(
			IdResultType idResultType,
			IdResult idResult,
			IdRef vector)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpAny, wordCount);
			writeWords(idResultType, idResult, vector);
		}

		void OpArbitraryFloatACosINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t M1,
			uint32_t mout,
			uint32_t enableSubnormals,
			uint32_t roundingMode,
			uint32_t roundingAccuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatACosINTEL, wordCount);
			writeWords(idResultType, idResult, A, M1, mout, enableSubnormals, roundingMode, roundingAccuracy);
		}

		void OpArbitraryFloatACosPiINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatACosPiINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatASinINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatASinINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatASinPiINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatASinPiINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatATan2INTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			IdRef B,
			uint32_t mb,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 11;

			writeWord(spv::Op::OpArbitraryFloatATan2INTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, B, mb, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatATanINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatATanINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatATanPiINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatATanPiINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatAddINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			IdRef B,
			uint32_t mb,
			uint32_t mResult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 11;

			writeWord(spv::Op::OpArbitraryFloatAddINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, B, mb, mResult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatCastFromIntINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t mresult,
			uint32_t fromSign,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatCastFromIntINTEL, wordCount);
			writeWords(idResultType, idResult, A, mresult, fromSign, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatCastINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatCastINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatCastToIntINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t toSign,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatCastToIntINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, toSign, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatCbrtINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatCbrtINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatCosINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatCosINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatCosPiINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatCosPiINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatDivINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			IdRef B,
			uint32_t mb,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 11;

			writeWord(spv::Op::OpArbitraryFloatDivINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, B, mb, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatEQINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			IdRef B,
			uint32_t mb)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpArbitraryFloatEQINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, B, mb);
		}

		void OpArbitraryFloatExp10INTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatExp10INTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatExp2INTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatExp2INTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatExpINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatExpINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatExpm1INTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatExpm1INTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatGEINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			IdRef B,
			uint32_t mb)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpArbitraryFloatGEINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, B, mb);
		}

		void OpArbitraryFloatGTINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			IdRef B,
			uint32_t mb)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpArbitraryFloatGTINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, B, mb);
		}

		void OpArbitraryFloatHypotINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			IdRef B,
			uint32_t mb,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 11;

			writeWord(spv::Op::OpArbitraryFloatHypotINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, B, mb, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatLEINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			IdRef B,
			uint32_t mb)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpArbitraryFloatLEINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, B, mb);
		}

		void OpArbitraryFloatLTINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			IdRef B,
			uint32_t mb)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpArbitraryFloatLTINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, B, mb);
		}

		void OpArbitraryFloatLog10INTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatLog10INTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatLog1pINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatLog1pINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatLog2INTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatLog2INTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatLogINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatLogINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatMulINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			IdRef B,
			uint32_t mb,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 11;

			writeWord(spv::Op::OpArbitraryFloatMulINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, B, mb, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatPowINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			IdRef B,
			uint32_t mb,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 11;

			writeWord(spv::Op::OpArbitraryFloatPowINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, B, mb, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatPowNINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			IdRef B,
			uint32_t signOfB,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 11;

			writeWord(spv::Op::OpArbitraryFloatPowNINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, B, signOfB, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatPowRINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			IdRef B,
			uint32_t mb,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 11;

			writeWord(spv::Op::OpArbitraryFloatPowRINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, B, mb, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatRSqrtINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatRSqrtINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatRecipINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatRecipINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatSinCosINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatSinCosINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatSinCosPiINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mResult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t roundingAccuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatSinCosPiINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mResult, subnormal, rounding, roundingAccuracy);
		}

		void OpArbitraryFloatSinINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatSinINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatSinPiINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatSinPiINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatSqrtINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpArbitraryFloatSqrtINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, mresult, subnormal, rounding, accuracy);
		}

		void OpArbitraryFloatSubINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			uint32_t ma,
			IdRef B,
			uint32_t mb,
			uint32_t mresult,
			uint32_t subnormal,
			uint32_t rounding,
			uint32_t accuracy)
		{
			uint16_t wordCount = 11;

			writeWord(spv::Op::OpArbitraryFloatSubINTEL, wordCount);
			writeWords(idResultType, idResult, A, ma, B, mb, mresult, subnormal, rounding, accuracy);
		}

		void OpArithmeticFenceEXT(
			IdResultType idResultType,
			IdResult idResult,
			IdRef target)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpArithmeticFenceEXT, wordCount);
			writeWords(idResultType, idResult, target);
		}

		void OpArrayLength(
			IdResultType idResultType,
			IdResult idResult,
			IdRef structure,
			uint32_t arrayMember)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpArrayLength, wordCount);
			writeWords(idResultType, idResult, structure, arrayMember);
		}

		void OpAsmCallINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef _asm,
			const std::vector<IdRef>& argument0 = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, argument0);

			writeWord(spv::Op::OpAsmCallINTEL, wordCount);
			writeWords(idResultType, idResult, _asm, argument0);
		}

		void OpAsmINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef asmType,
			IdRef target,
			const std::string& asmInstructions,
			const std::string& constraints)
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, asmInstructions, constraints);

			writeWord(spv::Op::OpAsmINTEL, wordCount);
			writeWords(idResultType, idResult, asmType, target, asmInstructions, constraints);
		}

		void OpAsmTargetINTEL(
			IdResult idResult,
			const std::string& asmTarget)
		{
			uint16_t wordCount = 2;
			countOperandsWord(wordCount, asmTarget);

			writeWord(spv::Op::OpAsmTargetINTEL, wordCount);
			writeWords(idResult, asmTarget);
		}

		void OpAssumeTrueKHR(IdRef condition)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpAssumeTrueKHR, wordCount);
			writeWords(condition);
		}

		void OpAtomicAnd(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdScope memory,
			IdMemorySemantics semantics,
			IdRef value)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpAtomicAnd, wordCount);
			writeWords(idResultType, idResult, pointer, memory, semantics, value);
		}

		void OpAtomicCompareExchange(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdScope memory,
			IdMemorySemantics equal,
			IdMemorySemantics unequal,
			IdRef value,
			IdRef comparator)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpAtomicCompareExchange, wordCount);
			writeWords(idResultType, idResult, pointer, memory, equal, unequal, value, comparator);
		}

		void OpAtomicCompareExchangeWeak(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdScope memory,
			IdMemorySemantics equal,
			IdMemorySemantics unequal,
			IdRef value,
			IdRef comparator)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpAtomicCompareExchangeWeak, wordCount);
			writeWords(idResultType, idResult, pointer, memory, equal, unequal, value, comparator);
		}

		void OpAtomicExchange(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdScope memory,
			IdMemorySemantics semantics,
			IdRef value)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpAtomicExchange, wordCount);
			writeWords(idResultType, idResult, pointer, memory, semantics, value);
		}

		void OpAtomicFAddEXT(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdScope memory,
			IdMemorySemantics semantics,
			IdRef value)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpAtomicFAddEXT, wordCount);
			writeWords(idResultType, idResult, pointer, memory, semantics, value);
		}

		void OpAtomicFMaxEXT(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdScope memory,
			IdMemorySemantics semantics,
			IdRef value)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpAtomicFMaxEXT, wordCount);
			writeWords(idResultType, idResult, pointer, memory, semantics, value);
		}

		void OpAtomicFMinEXT(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdScope memory,
			IdMemorySemantics semantics,
			IdRef value)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpAtomicFMinEXT, wordCount);
			writeWords(idResultType, idResult, pointer, memory, semantics, value);
		}

		void OpAtomicFlagClear(
			IdRef pointer,
			IdScope memory,
			IdMemorySemantics semantics)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpAtomicFlagClear, wordCount);
			writeWords(pointer, memory, semantics);
		}

		void OpAtomicFlagTestAndSet(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdScope memory,
			IdMemorySemantics semantics)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpAtomicFlagTestAndSet, wordCount);
			writeWords(idResultType, idResult, pointer, memory, semantics);
		}

		void OpAtomicIAdd(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdScope memory,
			IdMemorySemantics semantics,
			IdRef value)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpAtomicIAdd, wordCount);
			writeWords(idResultType, idResult, pointer, memory, semantics, value);
		}

		void OpAtomicIDecrement(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdScope memory,
			IdMemorySemantics semantics)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpAtomicIDecrement, wordCount);
			writeWords(idResultType, idResult, pointer, memory, semantics);
		}

		void OpAtomicIIncrement(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdScope memory,
			IdMemorySemantics semantics)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpAtomicIIncrement, wordCount);
			writeWords(idResultType, idResult, pointer, memory, semantics);
		}

		void OpAtomicISub(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdScope memory,
			IdMemorySemantics semantics,
			IdRef value)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpAtomicISub, wordCount);
			writeWords(idResultType, idResult, pointer, memory, semantics, value);
		}

		void OpAtomicLoad(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdScope memory,
			IdMemorySemantics semantics)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpAtomicLoad, wordCount);
			writeWords(idResultType, idResult, pointer, memory, semantics);
		}

		void OpAtomicOr(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdScope memory,
			IdMemorySemantics semantics,
			IdRef value)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpAtomicOr, wordCount);
			writeWords(idResultType, idResult, pointer, memory, semantics, value);
		}

		void OpAtomicSMax(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdScope memory,
			IdMemorySemantics semantics,
			IdRef value)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpAtomicSMax, wordCount);
			writeWords(idResultType, idResult, pointer, memory, semantics, value);
		}

		void OpAtomicSMin(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdScope memory,
			IdMemorySemantics semantics,
			IdRef value)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpAtomicSMin, wordCount);
			writeWords(idResultType, idResult, pointer, memory, semantics, value);
		}

		void OpAtomicStore(
			IdRef pointer,
			IdScope memory,
			IdMemorySemantics semantics,
			IdRef value)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpAtomicStore, wordCount);
			writeWords(pointer, memory, semantics, value);
		}

		void OpAtomicUMax(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdScope memory,
			IdMemorySemantics semantics,
			IdRef value)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpAtomicUMax, wordCount);
			writeWords(idResultType, idResult, pointer, memory, semantics, value);
		}

		void OpAtomicUMin(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdScope memory,
			IdMemorySemantics semantics,
			IdRef value)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpAtomicUMin, wordCount);
			writeWords(idResultType, idResult, pointer, memory, semantics, value);
		}

		void OpAtomicXor(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdScope memory,
			IdMemorySemantics semantics,
			IdRef value)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpAtomicXor, wordCount);
			writeWords(idResultType, idResult, pointer, memory, semantics, value);
		}

		void OpBeginInvocationInterlockEXT()
		{
			uint16_t wordCount = 1;

			writeWord(spv::Op::OpBeginInvocationInterlockEXT, wordCount);
			writeWords();
		}

		void OpBitCount(
			IdResultType idResultType,
			IdResult idResult,
			IdRef base)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpBitCount, wordCount);
			writeWords(idResultType, idResult, base);
		}

		void OpBitFieldInsert(
			IdResultType idResultType,
			IdResult idResult,
			IdRef base,
			IdRef insert,
			IdRef offset,
			IdRef count)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpBitFieldInsert, wordCount);
			writeWords(idResultType, idResult, base, insert, offset, count);
		}

		void OpBitFieldSExtract(
			IdResultType idResultType,
			IdResult idResult,
			IdRef base,
			IdRef offset,
			IdRef count)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpBitFieldSExtract, wordCount);
			writeWords(idResultType, idResult, base, offset, count);
		}

		void OpBitFieldUExtract(
			IdResultType idResultType,
			IdResult idResult,
			IdRef base,
			IdRef offset,
			IdRef count)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpBitFieldUExtract, wordCount);
			writeWords(idResultType, idResult, base, offset, count);
		}

		void OpBitReverse(
			IdResultType idResultType,
			IdResult idResult,
			IdRef base)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpBitReverse, wordCount);
			writeWords(idResultType, idResult, base);
		}

		void OpBitcast(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpBitcast, wordCount);
			writeWords(idResultType, idResult, operand);
		}

		void OpBitwiseAnd(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpBitwiseAnd, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpBitwiseFunctionINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			IdRef B,
			IdRef C,
			IdRef lUTIndex)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpBitwiseFunctionINTEL, wordCount);
			writeWords(idResultType, idResult, A, B, C, lUTIndex);
		}

		void OpBitwiseOr(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpBitwiseOr, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpBitwiseXor(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpBitwiseXor, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpBranch(IdRef targetLabel)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpBranch, wordCount);
			writeWords(targetLabel);
		}

		void OpBranchConditional(
			IdRef condition,
			IdRef trueLabel,
			IdRef falseLabel,
			const std::vector<uint32_t>& branchWeights = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, branchWeights);

			writeWord(spv::Op::OpBranchConditional, wordCount);
			writeWords(condition, trueLabel, falseLabel, branchWeights);
		}

		void OpBuildNDRange(
			IdResultType idResultType,
			IdResult idResult,
			IdRef globalWorkSize,
			IdRef localWorkSize,
			IdRef globalWorkOffset)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpBuildNDRange, wordCount);
			writeWords(idResultType, idResult, globalWorkSize, localWorkSize, globalWorkOffset);
		}

		void OpCapability(spv::Capability capability)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpCapability, wordCount);
			writeWords(capability);
		}

		void OpCaptureEventProfilingInfo(
			IdRef event,
			IdRef profilingInfo,
			IdRef value)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpCaptureEventProfilingInfo, wordCount);
			writeWords(event, profilingInfo, value);
		}

		void OpColorAttachmentReadEXT(
			IdResultType idResultType,
			IdResult idResult,
			IdRef attachment,
			std::optional<IdRef> sample = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, sample);

			writeWord(spv::Op::OpColorAttachmentReadEXT, wordCount);
			writeWords(idResultType, idResult, attachment, sample);
		}

		void OpCommitReadPipe(
			IdRef pipe,
			IdRef reserveId,
			IdRef packetSize,
			IdRef packetAlignment)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpCommitReadPipe, wordCount);
			writeWords(pipe, reserveId, packetSize, packetAlignment);
		}

		void OpCommitWritePipe(
			IdRef pipe,
			IdRef reserveId,
			IdRef packetSize,
			IdRef packetAlignment)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpCommitWritePipe, wordCount);
			writeWords(pipe, reserveId, packetSize, packetAlignment);
		}

		void OpCompositeConstruct(
			IdResultType idResultType,
			IdResult idResult,
			const std::vector<IdRef>& constituents = {})
		{
			uint16_t wordCount = 3;
			countOperandsWord(wordCount, constituents);

			writeWord(spv::Op::OpCompositeConstruct, wordCount);
			writeWords(idResultType, idResult, constituents);
		}

		void OpCompositeConstructContinuedINTEL(
			IdResultType idResultType,
			IdResult idResult,
			const std::vector<IdRef>& constituents = {})
		{
			uint16_t wordCount = 3;
			countOperandsWord(wordCount, constituents);

			writeWord(spv::Op::OpCompositeConstructContinuedINTEL, wordCount);
			writeWords(idResultType, idResult, constituents);
		}

		void OpCompositeConstructReplicateEXT(
			IdResultType idResultType,
			IdResult idResult,
			IdRef value)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpCompositeConstructReplicateEXT, wordCount);
			writeWords(idResultType, idResult, value);
		}

		void OpCompositeExtract(
			IdResultType idResultType,
			IdResult idResult,
			IdRef composite,
			const std::vector<uint32_t>& indexes = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, indexes);

			writeWord(spv::Op::OpCompositeExtract, wordCount);
			writeWords(idResultType, idResult, composite, indexes);
		}

		void OpCompositeInsert(
			IdResultType idResultType,
			IdResult idResult,
			IdRef object,
			IdRef composite,
			const std::vector<uint32_t>& indexes = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, indexes);

			writeWord(spv::Op::OpCompositeInsert, wordCount);
			writeWords(idResultType, idResult, object, composite, indexes);
		}

		void OpConstant(
			IdResultType idResultType,
			IdResult idResult,
			spvConstant auto value)
		{
			uint16_t wordCount = 3;
			countOperandsWord(wordCount, value);

			writeWord(spv::Op::OpConstant, wordCount);
			writeWords(idResultType, idResult, value);
		}

		void OpConstantComposite(
			IdResultType idResultType,
			IdResult idResult,
			const std::vector<IdRef>& constituents = {})
		{
			uint16_t wordCount = 3;
			countOperandsWord(wordCount, constituents);

			writeWord(spv::Op::OpConstantComposite, wordCount);
			writeWords(idResultType, idResult, constituents);
		}

		void OpConstantCompositeContinuedINTEL(const std::vector<IdRef>& constituents = {})
		{
			uint16_t wordCount = 1;
			countOperandsWord(wordCount, constituents);

			writeWord(spv::Op::OpConstantCompositeContinuedINTEL, wordCount);
			writeWords(constituents);
		}

		void OpConstantCompositeReplicateEXT(
			IdResultType idResultType,
			IdResult idResult,
			IdRef value)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpConstantCompositeReplicateEXT, wordCount);
			writeWords(idResultType, idResult, value);
		}

		void OpConstantFalse(
			IdResultType idResultType,
			IdResult idResult)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpConstantFalse, wordCount);
			writeWords(idResultType, idResult);
		}

		void OpConstantFunctionPointerINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef function)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpConstantFunctionPointerINTEL, wordCount);
			writeWords(idResultType, idResult, function);
		}

		void OpConstantNull(
			IdResultType idResultType,
			IdResult idResult)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpConstantNull, wordCount);
			writeWords(idResultType, idResult);
		}

		void OpConstantPipeStorage(
			IdResultType idResultType,
			IdResult idResult,
			uint32_t packetSize,
			uint32_t packetAlignment,
			uint32_t capacity)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpConstantPipeStorage, wordCount);
			writeWords(idResultType, idResult, packetSize, packetAlignment, capacity);
		}

		void OpConstantSampler(
			IdResultType idResultType,
			IdResult idResult,
			spv::SamplerAddressingMode samplerAddressingMode,
			uint32_t param,
			spv::SamplerFilterMode samplerFilterMode)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpConstantSampler, wordCount);
			writeWords(idResultType, idResult, samplerAddressingMode, param, samplerFilterMode);
		}

		void OpConstantStringAMDX(
			IdResult idResult,
			const std::string& literalString)
		{
			uint16_t wordCount = 2;
			countOperandsWord(wordCount, literalString);

			writeWord(spv::Op::OpConstantStringAMDX, wordCount);
			writeWords(idResult, literalString);
		}

		void OpConstantTrue(
			IdResultType idResultType,
			IdResult idResult)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpConstantTrue, wordCount);
			writeWords(idResultType, idResult);
		}

		void OpControlBarrier(
			IdScope execution,
			IdScope memory,
			IdMemorySemantics semantics)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpControlBarrier, wordCount);
			writeWords(execution, memory, semantics);
		}

		void OpControlBarrierArriveINTEL(
			IdScope execution,
			IdScope memory,
			IdMemorySemantics semantics)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpControlBarrierArriveINTEL, wordCount);
			writeWords(execution, memory, semantics);
		}

		void OpControlBarrierWaitINTEL(
			IdScope execution,
			IdScope memory,
			IdMemorySemantics semantics)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpControlBarrierWaitINTEL, wordCount);
			writeWords(execution, memory, semantics);
		}

		void OpConvertBF16ToFINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef bFloat16Value)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpConvertBF16ToFINTEL, wordCount);
			writeWords(idResultType, idResult, bFloat16Value);
		}

		void OpConvertFToBF16INTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef floatValue)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpConvertFToBF16INTEL, wordCount);
			writeWords(idResultType, idResult, floatValue);
		}

		void OpConvertFToS(
			IdResultType idResultType,
			IdResult idResult,
			IdRef floatValue)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpConvertFToS, wordCount);
			writeWords(idResultType, idResult, floatValue);
		}

		void OpConvertFToU(
			IdResultType idResultType,
			IdResult idResult,
			IdRef floatValue)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpConvertFToU, wordCount);
			writeWords(idResultType, idResult, floatValue);
		}

		void OpConvertImageToUNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpConvertImageToUNV, wordCount);
			writeWords(idResultType, idResult, operand);
		}

		void OpConvertPtrToU(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpConvertPtrToU, wordCount);
			writeWords(idResultType, idResult, pointer);
		}

		void OpConvertSToF(
			IdResultType idResultType,
			IdResult idResult,
			IdRef signedValue)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpConvertSToF, wordCount);
			writeWords(idResultType, idResult, signedValue);
		}

		void OpConvertSampledImageToUNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpConvertSampledImageToUNV, wordCount);
			writeWords(idResultType, idResult, operand);
		}

		void OpConvertSamplerToUNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpConvertSamplerToUNV, wordCount);
			writeWords(idResultType, idResult, operand);
		}

		void OpConvertUToAccelerationStructureKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef accel)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpConvertUToAccelerationStructureKHR, wordCount);
			writeWords(idResultType, idResult, accel);
		}

		void OpConvertUToF(
			IdResultType idResultType,
			IdResult idResult,
			IdRef unsignedValue)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpConvertUToF, wordCount);
			writeWords(idResultType, idResult, unsignedValue);
		}

		void OpConvertUToImageNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpConvertUToImageNV, wordCount);
			writeWords(idResultType, idResult, operand);
		}

		void OpConvertUToPtr(
			IdResultType idResultType,
			IdResult idResult,
			IdRef integerValue)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpConvertUToPtr, wordCount);
			writeWords(idResultType, idResult, integerValue);
		}

		void OpConvertUToSampledImageNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpConvertUToSampledImageNV, wordCount);
			writeWords(idResultType, idResult, operand);
		}

		void OpConvertUToSamplerNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpConvertUToSamplerNV, wordCount);
			writeWords(idResultType, idResult, operand);
		}

		void OpCooperativeMatrixConvertNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef matrix)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpCooperativeMatrixConvertNV, wordCount);
			writeWords(idResultType, idResult, matrix);
		}

		void OpCooperativeMatrixLengthKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef type)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpCooperativeMatrixLengthKHR, wordCount);
			writeWords(idResultType, idResult, type);
		}

		void OpCooperativeMatrixLengthNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef type)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpCooperativeMatrixLengthNV, wordCount);
			writeWords(idResultType, idResult, type);
		}

		void OpCooperativeMatrixLoadKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdRef memoryLayout,
			std::optional<IdRef> stride = {},
			std::optional<spv::MemoryAccessMask> memoryOperand = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, stride, memoryOperand);

			writeWord(spv::Op::OpCooperativeMatrixLoadKHR, wordCount);
			writeWords(idResultType, idResult, pointer, memoryLayout, stride, memoryOperand);
		}

		void OpCooperativeMatrixLoadNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdRef stride,
			IdRef columnMajor,
			std::optional<spv::MemoryAccessMask> memoryAccess = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, memoryAccess);

			writeWord(spv::Op::OpCooperativeMatrixLoadNV, wordCount);
			writeWords(idResultType, idResult, pointer, stride, columnMajor, memoryAccess);
		}

		void OpCooperativeMatrixLoadTensorNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdRef object,
			IdRef tensorLayout,
			spv::MemoryAccessMask memoryOperand,
			spv::TensorAddressingOperandsMask tensorAddressingOperands)
		{
			uint16_t wordCount = 8;

			writeWord(spv::Op::OpCooperativeMatrixLoadTensorNV, wordCount);
			writeWords(idResultType, idResult, pointer, object, tensorLayout, memoryOperand, tensorAddressingOperands);
		}

		void OpCooperativeMatrixMulAddKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			IdRef B,
			IdRef C,
			std::optional<spv::CooperativeMatrixOperandsMask> cooperativeMatrixOperands = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, cooperativeMatrixOperands);

			writeWord(spv::Op::OpCooperativeMatrixMulAddKHR, wordCount);
			writeWords(idResultType, idResult, A, B, C, cooperativeMatrixOperands);
		}

		void OpCooperativeMatrixMulAddNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef A,
			IdRef B,
			IdRef C)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpCooperativeMatrixMulAddNV, wordCount);
			writeWords(idResultType, idResult, A, B, C);
		}

		void OpCooperativeMatrixPerElementOpNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef matrix,
			IdRef func,
			const std::vector<IdRef>& operands = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, operands);

			writeWord(spv::Op::OpCooperativeMatrixPerElementOpNV, wordCount);
			writeWords(idResultType, idResult, matrix, func, operands);
		}

		void OpCooperativeMatrixReduceNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef matrix,
			spv::CooperativeMatrixReduceMask reduce,
			IdRef combineFunc)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpCooperativeMatrixReduceNV, wordCount);
			writeWords(idResultType, idResult, matrix, reduce, combineFunc);
		}

		void OpCooperativeMatrixStoreKHR(
			IdRef pointer,
			IdRef object,
			IdRef memoryLayout,
			std::optional<IdRef> stride = {},
			std::optional<spv::MemoryAccessMask> memoryOperand = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, stride, memoryOperand);

			writeWord(spv::Op::OpCooperativeMatrixStoreKHR, wordCount);
			writeWords(pointer, object, memoryLayout, stride, memoryOperand);
		}

		void OpCooperativeMatrixStoreNV(
			IdRef pointer,
			IdRef object,
			IdRef stride,
			IdRef columnMajor,
			std::optional<spv::MemoryAccessMask> memoryAccess = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, memoryAccess);

			writeWord(spv::Op::OpCooperativeMatrixStoreNV, wordCount);
			writeWords(pointer, object, stride, columnMajor, memoryAccess);
		}

		void OpCooperativeMatrixStoreTensorNV(
			IdRef pointer,
			IdRef object,
			IdRef tensorLayout,
			spv::MemoryAccessMask memoryOperand,
			spv::TensorAddressingOperandsMask tensorAddressingOperands)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpCooperativeMatrixStoreTensorNV, wordCount);
			writeWords(pointer, object, tensorLayout, memoryOperand, tensorAddressingOperands);
		}

		void OpCooperativeMatrixTransposeNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef matrix)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpCooperativeMatrixTransposeNV, wordCount);
			writeWords(idResultType, idResult, matrix);
		}

		void OpCooperativeVectorLoadNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			IdRef offset,
			std::optional<spv::MemoryAccessMask> memoryAccess = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, memoryAccess);

			writeWord(spv::Op::OpCooperativeVectorLoadNV, wordCount);
			writeWords(idResultType, idResult, pointer, offset, memoryAccess);
		}

		void OpCooperativeVectorMatrixMulAddNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef input,
			IdRef inputInterpretation,
			IdRef matrix,
			IdRef matrixOffset,
			IdRef matrixInterpretation,
			IdRef bias,
			IdRef biasOffset,
			IdRef biasInterpretation,
			IdRef M,
			IdRef K,
			IdRef memoryLayout,
			IdRef transpose,
			std::optional<IdRef> matrixStride = {},
			std::optional<spv::CooperativeMatrixOperandsMask> cooperativeMatrixOperands = {})
		{
			uint16_t wordCount = 15;
			countOperandsWord(wordCount, matrixStride, cooperativeMatrixOperands);

			writeWord(spv::Op::OpCooperativeVectorMatrixMulAddNV, wordCount);
			writeWords(idResultType, idResult, input, inputInterpretation, matrix, matrixOffset, matrixInterpretation, bias, biasOffset, biasInterpretation, M, K, memoryLayout, transpose, matrixStride, cooperativeMatrixOperands);
		}

		void OpCooperativeVectorMatrixMulNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef input,
			IdRef inputInterpretation,
			IdRef matrix,
			IdRef matrixOffset,
			IdRef matrixInterpretation,
			IdRef M,
			IdRef K,
			IdRef memoryLayout,
			IdRef transpose,
			std::optional<IdRef> matrixStride = {},
			std::optional<spv::CooperativeMatrixOperandsMask> cooperativeMatrixOperands = {})
		{
			uint16_t wordCount = 12;
			countOperandsWord(wordCount, matrixStride, cooperativeMatrixOperands);

			writeWord(spv::Op::OpCooperativeVectorMatrixMulNV, wordCount);
			writeWords(idResultType, idResult, input, inputInterpretation, matrix, matrixOffset, matrixInterpretation, M, K, memoryLayout, transpose, matrixStride, cooperativeMatrixOperands);
		}

		void OpCooperativeVectorOuterProductAccumulateNV(
			IdRef pointer,
			IdRef offset,
			IdRef A,
			IdRef B,
			IdRef memoryLayout,
			IdRef matrixInterpretation,
			std::optional<IdRef> matrixStride = {})
		{
			uint16_t wordCount = 7;
			countOperandsWord(wordCount, matrixStride);

			writeWord(spv::Op::OpCooperativeVectorOuterProductAccumulateNV, wordCount);
			writeWords(pointer, offset, A, B, memoryLayout, matrixInterpretation, matrixStride);
		}

		void OpCooperativeVectorReduceSumAccumulateNV(
			IdRef pointer,
			IdRef offset,
			IdRef V)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpCooperativeVectorReduceSumAccumulateNV, wordCount);
			writeWords(pointer, offset, V);
		}

		void OpCooperativeVectorStoreNV(
			IdRef pointer,
			IdRef offset,
			IdRef object,
			std::optional<spv::MemoryAccessMask> memoryAccess = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, memoryAccess);

			writeWord(spv::Op::OpCooperativeVectorStoreNV, wordCount);
			writeWords(pointer, offset, object, memoryAccess);
		}

		void OpCopyLogical(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpCopyLogical, wordCount);
			writeWords(idResultType, idResult, operand);
		}

		void OpCopyMemory(
			IdRef target,
			IdRef source,
			std::optional<spv::MemoryAccessMask> memoryAccess1 = {},
			std::optional<spv::MemoryAccessMask> memoryAccess2 = {})
		{
			uint16_t wordCount = 3;
			countOperandsWord(wordCount, memoryAccess1, memoryAccess2);

			writeWord(spv::Op::OpCopyMemory, wordCount);
			writeWords(target, source, memoryAccess1, memoryAccess2);
		}

		void OpCopyMemorySized(
			IdRef target,
			IdRef source,
			IdRef size,
			std::optional<spv::MemoryAccessMask> memoryAccess1 = {},
			std::optional<spv::MemoryAccessMask> memoryAccess2 = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, memoryAccess1, memoryAccess2);

			writeWord(spv::Op::OpCopyMemorySized, wordCount);
			writeWords(target, source, size, memoryAccess1, memoryAccess2);
		}

		void OpCopyObject(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpCopyObject, wordCount);
			writeWords(idResultType, idResult, operand);
		}

		void OpCreatePipeFromPipeStorage(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pipeStorage)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpCreatePipeFromPipeStorage, wordCount);
			writeWords(idResultType, idResult, pipeStorage);
		}

		void OpCreateTensorLayoutNV(
			IdResultType idResultType,
			IdResult idResult)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpCreateTensorLayoutNV, wordCount);
			writeWords(idResultType, idResult);
		}

		void OpCreateTensorViewNV(
			IdResultType idResultType,
			IdResult idResult)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpCreateTensorViewNV, wordCount);
			writeWords(idResultType, idResult);
		}

		void OpCreateUserEvent(
			IdResultType idResultType,
			IdResult idResult)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpCreateUserEvent, wordCount);
			writeWords(idResultType, idResult);
		}

		void OpCrossWorkgroupCastToPtrINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpCrossWorkgroupCastToPtrINTEL, wordCount);
			writeWords(idResultType, idResult, pointer);
		}

		void OpDPdx(
			IdResultType idResultType,
			IdResult idResult,
			IdRef P)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpDPdx, wordCount);
			writeWords(idResultType, idResult, P);
		}

		void OpDPdxCoarse(
			IdResultType idResultType,
			IdResult idResult,
			IdRef P)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpDPdxCoarse, wordCount);
			writeWords(idResultType, idResult, P);
		}

		void OpDPdxFine(
			IdResultType idResultType,
			IdResult idResult,
			IdRef P)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpDPdxFine, wordCount);
			writeWords(idResultType, idResult, P);
		}

		void OpDPdy(
			IdResultType idResultType,
			IdResult idResult,
			IdRef P)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpDPdy, wordCount);
			writeWords(idResultType, idResult, P);
		}

		void OpDPdyCoarse(
			IdResultType idResultType,
			IdResult idResult,
			IdRef P)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpDPdyCoarse, wordCount);
			writeWords(idResultType, idResult, P);
		}

		void OpDPdyFine(
			IdResultType idResultType,
			IdResult idResult,
			IdRef P)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpDPdyFine, wordCount);
			writeWords(idResultType, idResult, P);
		}

		void OpDecorate(
			IdRef target,
			spv::Decoration decoration)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpDecorate, wordCount);
			writeWords(target, decoration);
		}

		void OpDecorateId(
			IdRef target,
			spv::Decoration decoration)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpDecorateId, wordCount);
			writeWords(target, decoration);
		}

		void OpDecorateString(
			IdRef target,
			spv::Decoration decoration)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpDecorateString, wordCount);
			writeWords(target, decoration);
		}

		void OpDecorationGroup(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpDecorationGroup, wordCount);
			writeWords(idResult);
		}

		void OpDemoteToHelperInvocation()
		{
			uint16_t wordCount = 1;

			writeWord(spv::Op::OpDemoteToHelperInvocation, wordCount);
			writeWords();
		}

		void OpDepthAttachmentReadEXT(
			IdResultType idResultType,
			IdResult idResult,
			std::optional<IdRef> sample = {})
		{
			uint16_t wordCount = 3;
			countOperandsWord(wordCount, sample);

			writeWord(spv::Op::OpDepthAttachmentReadEXT, wordCount);
			writeWords(idResultType, idResult, sample);
		}

		void OpDot(
			IdResultType idResultType,
			IdResult idResult,
			IdRef vector1,
			IdRef vector2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpDot, wordCount);
			writeWords(idResultType, idResult, vector1, vector2);
		}

		void OpEmitMeshTasksEXT(
			IdRef groupCountX,
			IdRef groupCountY,
			IdRef groupCountZ,
			std::optional<IdRef> payload = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, payload);

			writeWord(spv::Op::OpEmitMeshTasksEXT, wordCount);
			writeWords(groupCountX, groupCountY, groupCountZ, payload);
		}

		void OpEmitStreamVertex(IdRef stream)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpEmitStreamVertex, wordCount);
			writeWords(stream);
		}

		void OpEmitVertex()
		{
			uint16_t wordCount = 1;

			writeWord(spv::Op::OpEmitVertex, wordCount);
			writeWords();
		}

		void OpEndInvocationInterlockEXT()
		{
			uint16_t wordCount = 1;

			writeWord(spv::Op::OpEndInvocationInterlockEXT, wordCount);
			writeWords();
		}

		void OpEndPrimitive()
		{
			uint16_t wordCount = 1;

			writeWord(spv::Op::OpEndPrimitive, wordCount);
			writeWords();
		}

		void OpEndStreamPrimitive(IdRef stream)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpEndStreamPrimitive, wordCount);
			writeWords(stream);
		}

		void OpEnqueueKernel(
			IdResultType idResultType,
			IdResult idResult,
			IdRef queue,
			IdRef flags,
			IdRef nDRange,
			IdRef numEvents,
			IdRef waitEvents,
			IdRef retEvent,
			IdRef invoke,
			IdRef param,
			IdRef paramSize,
			IdRef paramAlign,
			const std::vector<IdRef>& localSize = {})
		{
			uint16_t wordCount = 13;
			countOperandsWord(wordCount, localSize);

			writeWord(spv::Op::OpEnqueueKernel, wordCount);
			writeWords(idResultType, idResult, queue, flags, nDRange, numEvents, waitEvents, retEvent, invoke, param, paramSize, paramAlign, localSize);
		}

		void OpEnqueueMarker(
			IdResultType idResultType,
			IdResult idResult,
			IdRef queue,
			IdRef numEvents,
			IdRef waitEvents,
			IdRef retEvent)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpEnqueueMarker, wordCount);
			writeWords(idResultType, idResult, queue, numEvents, waitEvents, retEvent);
		}

		void OpEnqueueNodePayloadsAMDX(IdRef payloadArray)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpEnqueueNodePayloadsAMDX, wordCount);
			writeWords(payloadArray);
		}

		void OpEntryPoint(
			spv::ExecutionModel executionModel,
			IdRef entryPoint,
			const std::string& name,
			const std::vector<IdRef>& interface = {})
		{
			uint16_t wordCount = 3;
			countOperandsWord(wordCount, name, interface);

			writeWord(spv::Op::OpEntryPoint, wordCount);
			writeWords(executionModel, entryPoint, name, interface);
		}

		void OpExecuteCallableKHR(
			IdRef sBTIndex,
			IdRef callableData)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpExecuteCallableKHR, wordCount);
			writeWords(sBTIndex, callableData);
		}

		void OpExecuteCallableNV(
			IdRef sBTIndex,
			IdRef callableDataId)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpExecuteCallableNV, wordCount);
			writeWords(sBTIndex, callableDataId);
		}

		void OpExecutionMode(
			IdRef entryPoint,
			spv::ExecutionMode mode)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpExecutionMode, wordCount);
			writeWords(entryPoint, mode);
		}

		void OpExecutionModeId(
			IdRef entryPoint,
			spv::ExecutionMode mode)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpExecutionModeId, wordCount);
			writeWords(entryPoint, mode);
		}

		void OpExpectKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef value,
			IdRef expectedValue)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpExpectKHR, wordCount);
			writeWords(idResultType, idResult, value, expectedValue);
		}

		void OpExtInst(
			IdResultType idResultType,
			IdResult idResult,
			IdRef set,
			uint32_t instruction,
			const std::vector<IdRef>& operands = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, operands);

			writeWord(spv::Op::OpExtInst, wordCount);
			writeWords(idResultType, idResult, set, instruction, operands);
		}

		void OpExtInstImport(
			IdResult idResult,
			const std::string& name)
		{
			uint16_t wordCount = 2;
			countOperandsWord(wordCount, name);

			writeWord(spv::Op::OpExtInstImport, wordCount);
			writeWords(idResult, name);
		}

		void OpExtInstWithForwardRefsKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef set,
			uint32_t instruction,
			const std::vector<IdRef>& operands = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, operands);

			writeWord(spv::Op::OpExtInstWithForwardRefsKHR, wordCount);
			writeWords(idResultType, idResult, set, instruction, operands);
		}

		void OpExtension(const std::string& name)
		{
			uint16_t wordCount = 1;
			countOperandsWord(wordCount, name);

			writeWord(spv::Op::OpExtension, wordCount);
			writeWords(name);
		}

		void OpFAdd(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpFAdd, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpFConvert(
			IdResultType idResultType,
			IdResult idResult,
			IdRef floatValue)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpFConvert, wordCount);
			writeWords(idResultType, idResult, floatValue);
		}

		void OpFDiv(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpFDiv, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpFMod(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpFMod, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpFMul(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpFMul, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpFNegate(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpFNegate, wordCount);
			writeWords(idResultType, idResult, operand);
		}

		void OpFOrdEqual(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpFOrdEqual, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpFOrdGreaterThan(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpFOrdGreaterThan, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpFOrdGreaterThanEqual(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpFOrdGreaterThanEqual, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpFOrdLessThan(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpFOrdLessThan, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpFOrdLessThanEqual(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpFOrdLessThanEqual, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpFOrdNotEqual(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpFOrdNotEqual, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpFPGARegINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef input)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpFPGARegINTEL, wordCount);
			writeWords(idResultType, idResult, input);
		}

		void OpFRem(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpFRem, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpFSub(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpFSub, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpFUnordEqual(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpFUnordEqual, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpFUnordGreaterThan(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpFUnordGreaterThan, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpFUnordGreaterThanEqual(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpFUnordGreaterThanEqual, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpFUnordLessThan(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpFUnordLessThan, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpFUnordLessThanEqual(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpFUnordLessThanEqual, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpFUnordNotEqual(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpFUnordNotEqual, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpFetchMicroTriangleVertexBarycentricNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef accel,
			IdRef instanceId,
			IdRef geometryIndex,
			IdRef primitiveIndex,
			IdRef barycentric)
		{
			uint16_t wordCount = 8;

			writeWord(spv::Op::OpFetchMicroTriangleVertexBarycentricNV, wordCount);
			writeWords(idResultType, idResult, accel, instanceId, geometryIndex, primitiveIndex, barycentric);
		}

		void OpFetchMicroTriangleVertexPositionNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef accel,
			IdRef instanceId,
			IdRef geometryIndex,
			IdRef primitiveIndex,
			IdRef barycentric)
		{
			uint16_t wordCount = 8;

			writeWord(spv::Op::OpFetchMicroTriangleVertexPositionNV, wordCount);
			writeWords(idResultType, idResult, accel, instanceId, geometryIndex, primitiveIndex, barycentric);
		}

		void OpFinishWritingNodePayloadAMDX(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpFinishWritingNodePayloadAMDX, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpFixedCosINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef input,
			uint32_t S,
			uint32_t I,
			uint32_t rI,
			uint32_t Q,
			uint32_t O)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpFixedCosINTEL, wordCount);
			writeWords(idResultType, idResult, input, S, I, rI, Q, O);
		}

		void OpFixedCosPiINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef input,
			uint32_t S,
			uint32_t I,
			uint32_t rI,
			uint32_t Q,
			uint32_t O)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpFixedCosPiINTEL, wordCount);
			writeWords(idResultType, idResult, input, S, I, rI, Q, O);
		}

		void OpFixedExpINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef input,
			uint32_t S,
			uint32_t I,
			uint32_t rI,
			uint32_t Q,
			uint32_t O)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpFixedExpINTEL, wordCount);
			writeWords(idResultType, idResult, input, S, I, rI, Q, O);
		}

		void OpFixedLogINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef input,
			uint32_t S,
			uint32_t I,
			uint32_t rI,
			uint32_t Q,
			uint32_t O)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpFixedLogINTEL, wordCount);
			writeWords(idResultType, idResult, input, S, I, rI, Q, O);
		}

		void OpFixedRecipINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef input,
			uint32_t S,
			uint32_t I,
			uint32_t rI,
			uint32_t Q,
			uint32_t O)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpFixedRecipINTEL, wordCount);
			writeWords(idResultType, idResult, input, S, I, rI, Q, O);
		}

		void OpFixedRsqrtINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef input,
			uint32_t S,
			uint32_t I,
			uint32_t rI,
			uint32_t Q,
			uint32_t O)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpFixedRsqrtINTEL, wordCount);
			writeWords(idResultType, idResult, input, S, I, rI, Q, O);
		}

		void OpFixedSinCosINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef input,
			uint32_t S,
			uint32_t I,
			uint32_t rI,
			uint32_t Q,
			uint32_t O)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpFixedSinCosINTEL, wordCount);
			writeWords(idResultType, idResult, input, S, I, rI, Q, O);
		}

		void OpFixedSinCosPiINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef input,
			uint32_t S,
			uint32_t I,
			uint32_t rI,
			uint32_t Q,
			uint32_t O)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpFixedSinCosPiINTEL, wordCount);
			writeWords(idResultType, idResult, input, S, I, rI, Q, O);
		}

		void OpFixedSinINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef input,
			uint32_t S,
			uint32_t I,
			uint32_t rI,
			uint32_t Q,
			uint32_t O)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpFixedSinINTEL, wordCount);
			writeWords(idResultType, idResult, input, S, I, rI, Q, O);
		}

		void OpFixedSinPiINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef input,
			uint32_t S,
			uint32_t I,
			uint32_t rI,
			uint32_t Q,
			uint32_t O)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpFixedSinPiINTEL, wordCount);
			writeWords(idResultType, idResult, input, S, I, rI, Q, O);
		}

		void OpFixedSqrtINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef input,
			uint32_t S,
			uint32_t I,
			uint32_t rI,
			uint32_t Q,
			uint32_t O)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpFixedSqrtINTEL, wordCount);
			writeWords(idResultType, idResult, input, S, I, rI, Q, O);
		}

		void OpFragmentFetchAMD(
			IdResultType idResultType,
			IdResult idResult,
			IdRef image,
			IdRef coordinate,
			IdRef fragmentIndex)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpFragmentFetchAMD, wordCount);
			writeWords(idResultType, idResult, image, coordinate, fragmentIndex);
		}

		void OpFragmentMaskFetchAMD(
			IdResultType idResultType,
			IdResult idResult,
			IdRef image,
			IdRef coordinate)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpFragmentMaskFetchAMD, wordCount);
			writeWords(idResultType, idResult, image, coordinate);
		}

		void OpFunction(
			IdResultType idResultType,
			IdResult idResult,
			spv::FunctionControlMask functionControl,
			IdRef functionType)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpFunction, wordCount);
			writeWords(idResultType, idResult, functionControl, functionType);
		}

		void OpFunctionCall(
			IdResultType idResultType,
			IdResult idResult,
			IdRef function,
			const std::vector<IdRef>& arguments = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, arguments);

			writeWord(spv::Op::OpFunctionCall, wordCount);
			writeWords(idResultType, idResult, function, arguments);
		}

		void OpFunctionEnd()
		{
			uint16_t wordCount = 1;

			writeWord(spv::Op::OpFunctionEnd, wordCount);
			writeWords();
		}

		void OpFunctionParameter(
			IdResultType idResultType,
			IdResult idResult)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpFunctionParameter, wordCount);
			writeWords(idResultType, idResult);
		}

		void OpFunctionPointerCallINTEL(
			IdResultType idResultType,
			IdResult idResult,
			const std::vector<IdRef>& operand1 = {})
		{
			uint16_t wordCount = 3;
			countOperandsWord(wordCount, operand1);

			writeWord(spv::Op::OpFunctionPointerCallINTEL, wordCount);
			writeWords(idResultType, idResult, operand1);
		}

		void OpFwidth(
			IdResultType idResultType,
			IdResult idResult,
			IdRef P)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpFwidth, wordCount);
			writeWords(idResultType, idResult, P);
		}

		void OpFwidthCoarse(
			IdResultType idResultType,
			IdResult idResult,
			IdRef P)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpFwidthCoarse, wordCount);
			writeWords(idResultType, idResult, P);
		}

		void OpFwidthFine(
			IdResultType idResultType,
			IdResult idResult,
			IdRef P)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpFwidthFine, wordCount);
			writeWords(idResultType, idResult, P);
		}

		void OpGenericCastToPtr(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpGenericCastToPtr, wordCount);
			writeWords(idResultType, idResult, pointer);
		}

		void OpGenericCastToPtrExplicit(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			spv::StorageClass storage)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpGenericCastToPtrExplicit, wordCount);
			writeWords(idResultType, idResult, pointer, storage);
		}

		void OpGenericPtrMemSemantics(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpGenericPtrMemSemantics, wordCount);
			writeWords(idResultType, idResult, pointer);
		}

		void OpGetDefaultQueue(
			IdResultType idResultType,
			IdResult idResult)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpGetDefaultQueue, wordCount);
			writeWords(idResultType, idResult);
		}

		void OpGetKernelLocalSizeForSubgroupCount(
			IdResultType idResultType,
			IdResult idResult,
			IdRef subgroupCount,
			IdRef invoke,
			IdRef param,
			IdRef paramSize,
			IdRef paramAlign)
		{
			uint16_t wordCount = 8;

			writeWord(spv::Op::OpGetKernelLocalSizeForSubgroupCount, wordCount);
			writeWords(idResultType, idResult, subgroupCount, invoke, param, paramSize, paramAlign);
		}

		void OpGetKernelMaxNumSubgroups(
			IdResultType idResultType,
			IdResult idResult,
			IdRef invoke,
			IdRef param,
			IdRef paramSize,
			IdRef paramAlign)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpGetKernelMaxNumSubgroups, wordCount);
			writeWords(idResultType, idResult, invoke, param, paramSize, paramAlign);
		}

		void OpGetKernelNDrangeMaxSubGroupSize(
			IdResultType idResultType,
			IdResult idResult,
			IdRef nDRange,
			IdRef invoke,
			IdRef param,
			IdRef paramSize,
			IdRef paramAlign)
		{
			uint16_t wordCount = 8;

			writeWord(spv::Op::OpGetKernelNDrangeMaxSubGroupSize, wordCount);
			writeWords(idResultType, idResult, nDRange, invoke, param, paramSize, paramAlign);
		}

		void OpGetKernelNDrangeSubGroupCount(
			IdResultType idResultType,
			IdResult idResult,
			IdRef nDRange,
			IdRef invoke,
			IdRef param,
			IdRef paramSize,
			IdRef paramAlign)
		{
			uint16_t wordCount = 8;

			writeWord(spv::Op::OpGetKernelNDrangeSubGroupCount, wordCount);
			writeWords(idResultType, idResult, nDRange, invoke, param, paramSize, paramAlign);
		}

		void OpGetKernelPreferredWorkGroupSizeMultiple(
			IdResultType idResultType,
			IdResult idResult,
			IdRef invoke,
			IdRef param,
			IdRef paramSize,
			IdRef paramAlign)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpGetKernelPreferredWorkGroupSizeMultiple, wordCount);
			writeWords(idResultType, idResult, invoke, param, paramSize, paramAlign);
		}

		void OpGetKernelWorkGroupSize(
			IdResultType idResultType,
			IdResult idResult,
			IdRef invoke,
			IdRef param,
			IdRef paramSize,
			IdRef paramAlign)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpGetKernelWorkGroupSize, wordCount);
			writeWords(idResultType, idResult, invoke, param, paramSize, paramAlign);
		}

		void OpGetMaxPipePackets(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pipe,
			IdRef packetSize,
			IdRef packetAlignment)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGetMaxPipePackets, wordCount);
			writeWords(idResultType, idResult, pipe, packetSize, packetAlignment);
		}

		void OpGetNumPipePackets(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pipe,
			IdRef packetSize,
			IdRef packetAlignment)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGetNumPipePackets, wordCount);
			writeWords(idResultType, idResult, pipe, packetSize, packetAlignment);
		}

		void OpGroupAll(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef predicate)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpGroupAll, wordCount);
			writeWords(idResultType, idResult, execution, predicate);
		}

		void OpGroupAny(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef predicate)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpGroupAny, wordCount);
			writeWords(idResultType, idResult, execution, predicate);
		}

		void OpGroupAsyncCopy(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef destination,
			IdRef source,
			IdRef numElements,
			IdRef stride,
			IdRef event)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpGroupAsyncCopy, wordCount);
			writeWords(idResultType, idResult, execution, destination, source, numElements, stride, event);
		}

		void OpGroupBitwiseAndKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupBitwiseAndKHR, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupBitwiseOrKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupBitwiseOrKHR, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupBitwiseXorKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupBitwiseXorKHR, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupBroadcast(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef value,
			IdRef localId)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupBroadcast, wordCount);
			writeWords(idResultType, idResult, execution, value, localId);
		}

		void OpGroupCommitReadPipe(
			IdScope execution,
			IdRef pipe,
			IdRef reserveId,
			IdRef packetSize,
			IdRef packetAlignment)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupCommitReadPipe, wordCount);
			writeWords(execution, pipe, reserveId, packetSize, packetAlignment);
		}

		void OpGroupCommitWritePipe(
			IdScope execution,
			IdRef pipe,
			IdRef reserveId,
			IdRef packetSize,
			IdRef packetAlignment)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupCommitWritePipe, wordCount);
			writeWords(execution, pipe, reserveId, packetSize, packetAlignment);
		}

		void OpGroupDecorate(
			IdRef decorationGroup,
			const std::vector<IdRef>& targets = {})
		{
			uint16_t wordCount = 2;
			countOperandsWord(wordCount, targets);

			writeWord(spv::Op::OpGroupDecorate, wordCount);
			writeWords(decorationGroup, targets);
		}

		void OpGroupFAdd(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupFAdd, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupFAddNonUniformAMD(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupFAddNonUniformAMD, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupFMax(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupFMax, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupFMaxNonUniformAMD(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupFMaxNonUniformAMD, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupFMin(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupFMin, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupFMinNonUniformAMD(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupFMinNonUniformAMD, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupFMulKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupFMulKHR, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupIAdd(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupIAdd, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupIAddNonUniformAMD(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupIAddNonUniformAMD, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupIMulKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupIMulKHR, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupLogicalAndKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupLogicalAndKHR, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupLogicalOrKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupLogicalOrKHR, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupLogicalXorKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupLogicalXorKHR, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupMemberDecorate(
			IdRef decorationGroup,
			const std::vector<std::tuple<IdRef, uint32_t>>& targets = {})
		{
			uint16_t wordCount = 2;
			countOperandsWord(wordCount, targets);

			writeWord(spv::Op::OpGroupMemberDecorate, wordCount);
			writeWords(decorationGroup, targets);
		}

		void OpGroupNonUniformAll(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef predicate)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpGroupNonUniformAll, wordCount);
			writeWords(idResultType, idResult, execution, predicate);
		}

		void OpGroupNonUniformAllEqual(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef value)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpGroupNonUniformAllEqual, wordCount);
			writeWords(idResultType, idResult, execution, value);
		}

		void OpGroupNonUniformAny(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef predicate)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpGroupNonUniformAny, wordCount);
			writeWords(idResultType, idResult, execution, predicate);
		}

		void OpGroupNonUniformBallot(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef predicate)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpGroupNonUniformBallot, wordCount);
			writeWords(idResultType, idResult, execution, predicate);
		}

		void OpGroupNonUniformBallotBitCount(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef value)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupNonUniformBallotBitCount, wordCount);
			writeWords(idResultType, idResult, execution, operation, value);
		}

		void OpGroupNonUniformBallotBitExtract(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef value,
			IdRef index)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupNonUniformBallotBitExtract, wordCount);
			writeWords(idResultType, idResult, execution, value, index);
		}

		void OpGroupNonUniformBallotFindLSB(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef value)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpGroupNonUniformBallotFindLSB, wordCount);
			writeWords(idResultType, idResult, execution, value);
		}

		void OpGroupNonUniformBallotFindMSB(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef value)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpGroupNonUniformBallotFindMSB, wordCount);
			writeWords(idResultType, idResult, execution, value);
		}

		void OpGroupNonUniformBitwiseAnd(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef value,
			std::optional<IdRef> clusterSize = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, clusterSize);

			writeWord(spv::Op::OpGroupNonUniformBitwiseAnd, wordCount);
			writeWords(idResultType, idResult, execution, operation, value, clusterSize);
		}

		void OpGroupNonUniformBitwiseOr(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef value,
			std::optional<IdRef> clusterSize = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, clusterSize);

			writeWord(spv::Op::OpGroupNonUniformBitwiseOr, wordCount);
			writeWords(idResultType, idResult, execution, operation, value, clusterSize);
		}

		void OpGroupNonUniformBitwiseXor(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef value,
			std::optional<IdRef> clusterSize = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, clusterSize);

			writeWord(spv::Op::OpGroupNonUniformBitwiseXor, wordCount);
			writeWords(idResultType, idResult, execution, operation, value, clusterSize);
		}

		void OpGroupNonUniformBroadcast(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef value,
			IdRef id)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupNonUniformBroadcast, wordCount);
			writeWords(idResultType, idResult, execution, value, id);
		}

		void OpGroupNonUniformBroadcastFirst(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef value)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpGroupNonUniformBroadcastFirst, wordCount);
			writeWords(idResultType, idResult, execution, value);
		}

		void OpGroupNonUniformElect(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpGroupNonUniformElect, wordCount);
			writeWords(idResultType, idResult, execution);
		}

		void OpGroupNonUniformFAdd(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef value,
			std::optional<IdRef> clusterSize = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, clusterSize);

			writeWord(spv::Op::OpGroupNonUniformFAdd, wordCount);
			writeWords(idResultType, idResult, execution, operation, value, clusterSize);
		}

		void OpGroupNonUniformFMax(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef value,
			std::optional<IdRef> clusterSize = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, clusterSize);

			writeWord(spv::Op::OpGroupNonUniformFMax, wordCount);
			writeWords(idResultType, idResult, execution, operation, value, clusterSize);
		}

		void OpGroupNonUniformFMin(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef value,
			std::optional<IdRef> clusterSize = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, clusterSize);

			writeWord(spv::Op::OpGroupNonUniformFMin, wordCount);
			writeWords(idResultType, idResult, execution, operation, value, clusterSize);
		}

		void OpGroupNonUniformFMul(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef value,
			std::optional<IdRef> clusterSize = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, clusterSize);

			writeWord(spv::Op::OpGroupNonUniformFMul, wordCount);
			writeWords(idResultType, idResult, execution, operation, value, clusterSize);
		}

		void OpGroupNonUniformIAdd(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef value,
			std::optional<IdRef> clusterSize = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, clusterSize);

			writeWord(spv::Op::OpGroupNonUniformIAdd, wordCount);
			writeWords(idResultType, idResult, execution, operation, value, clusterSize);
		}

		void OpGroupNonUniformIMul(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef value,
			std::optional<IdRef> clusterSize = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, clusterSize);

			writeWord(spv::Op::OpGroupNonUniformIMul, wordCount);
			writeWords(idResultType, idResult, execution, operation, value, clusterSize);
		}

		void OpGroupNonUniformInverseBallot(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef value)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpGroupNonUniformInverseBallot, wordCount);
			writeWords(idResultType, idResult, execution, value);
		}

		void OpGroupNonUniformLogicalAnd(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef value,
			std::optional<IdRef> clusterSize = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, clusterSize);

			writeWord(spv::Op::OpGroupNonUniformLogicalAnd, wordCount);
			writeWords(idResultType, idResult, execution, operation, value, clusterSize);
		}

		void OpGroupNonUniformLogicalOr(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef value,
			std::optional<IdRef> clusterSize = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, clusterSize);

			writeWord(spv::Op::OpGroupNonUniformLogicalOr, wordCount);
			writeWords(idResultType, idResult, execution, operation, value, clusterSize);
		}

		void OpGroupNonUniformLogicalXor(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef value,
			std::optional<IdRef> clusterSize = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, clusterSize);

			writeWord(spv::Op::OpGroupNonUniformLogicalXor, wordCount);
			writeWords(idResultType, idResult, execution, operation, value, clusterSize);
		}

		void OpGroupNonUniformPartitionNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef value)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpGroupNonUniformPartitionNV, wordCount);
			writeWords(idResultType, idResult, value);
		}

		void OpGroupNonUniformQuadAllKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef predicate)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpGroupNonUniformQuadAllKHR, wordCount);
			writeWords(idResultType, idResult, predicate);
		}

		void OpGroupNonUniformQuadAnyKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef predicate)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpGroupNonUniformQuadAnyKHR, wordCount);
			writeWords(idResultType, idResult, predicate);
		}

		void OpGroupNonUniformQuadBroadcast(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef value,
			IdRef index)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupNonUniformQuadBroadcast, wordCount);
			writeWords(idResultType, idResult, execution, value, index);
		}

		void OpGroupNonUniformQuadSwap(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef value,
			IdRef direction)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupNonUniformQuadSwap, wordCount);
			writeWords(idResultType, idResult, execution, value, direction);
		}

		void OpGroupNonUniformRotateKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef value,
			IdRef delta,
			std::optional<IdRef> clusterSize = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, clusterSize);

			writeWord(spv::Op::OpGroupNonUniformRotateKHR, wordCount);
			writeWords(idResultType, idResult, execution, value, delta, clusterSize);
		}

		void OpGroupNonUniformSMax(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef value,
			std::optional<IdRef> clusterSize = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, clusterSize);

			writeWord(spv::Op::OpGroupNonUniformSMax, wordCount);
			writeWords(idResultType, idResult, execution, operation, value, clusterSize);
		}

		void OpGroupNonUniformSMin(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef value,
			std::optional<IdRef> clusterSize = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, clusterSize);

			writeWord(spv::Op::OpGroupNonUniformSMin, wordCount);
			writeWords(idResultType, idResult, execution, operation, value, clusterSize);
		}

		void OpGroupNonUniformShuffle(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef value,
			IdRef id)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupNonUniformShuffle, wordCount);
			writeWords(idResultType, idResult, execution, value, id);
		}

		void OpGroupNonUniformShuffleDown(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef value,
			IdRef delta)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupNonUniformShuffleDown, wordCount);
			writeWords(idResultType, idResult, execution, value, delta);
		}

		void OpGroupNonUniformShuffleUp(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef value,
			IdRef delta)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupNonUniformShuffleUp, wordCount);
			writeWords(idResultType, idResult, execution, value, delta);
		}

		void OpGroupNonUniformShuffleXor(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef value,
			IdRef mask)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupNonUniformShuffleXor, wordCount);
			writeWords(idResultType, idResult, execution, value, mask);
		}

		void OpGroupNonUniformUMax(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef value,
			std::optional<IdRef> clusterSize = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, clusterSize);

			writeWord(spv::Op::OpGroupNonUniformUMax, wordCount);
			writeWords(idResultType, idResult, execution, operation, value, clusterSize);
		}

		void OpGroupNonUniformUMin(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef value,
			std::optional<IdRef> clusterSize = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, clusterSize);

			writeWord(spv::Op::OpGroupNonUniformUMin, wordCount);
			writeWords(idResultType, idResult, execution, operation, value, clusterSize);
		}

		void OpGroupReserveReadPipePackets(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef pipe,
			IdRef numPackets,
			IdRef packetSize,
			IdRef packetAlignment)
		{
			uint16_t wordCount = 8;

			writeWord(spv::Op::OpGroupReserveReadPipePackets, wordCount);
			writeWords(idResultType, idResult, execution, pipe, numPackets, packetSize, packetAlignment);
		}

		void OpGroupReserveWritePipePackets(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			IdRef pipe,
			IdRef numPackets,
			IdRef packetSize,
			IdRef packetAlignment)
		{
			uint16_t wordCount = 8;

			writeWord(spv::Op::OpGroupReserveWritePipePackets, wordCount);
			writeWords(idResultType, idResult, execution, pipe, numPackets, packetSize, packetAlignment);
		}

		void OpGroupSMax(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupSMax, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupSMaxNonUniformAMD(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupSMaxNonUniformAMD, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupSMin(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupSMin, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupSMinNonUniformAMD(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupSMinNonUniformAMD, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupUMax(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupUMax, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupUMaxNonUniformAMD(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupUMaxNonUniformAMD, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupUMin(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupUMin, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupUMinNonUniformAMD(
			IdResultType idResultType,
			IdResult idResult,
			IdScope execution,
			spv::GroupOperation operation,
			IdRef X)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpGroupUMinNonUniformAMD, wordCount);
			writeWords(idResultType, idResult, execution, operation, X);
		}

		void OpGroupWaitEvents(
			IdScope execution,
			IdRef numEvents,
			IdRef eventsList)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpGroupWaitEvents, wordCount);
			writeWords(execution, numEvents, eventsList);
		}

		void OpHitObjectExecuteShaderNV(
			IdRef hitObject,
			IdRef payload)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpHitObjectExecuteShaderNV, wordCount);
			writeWords(hitObject, payload);
		}

		void OpHitObjectGetAttributesNV(
			IdRef hitObject,
			IdRef hitObjectAttribute)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpHitObjectGetAttributesNV, wordCount);
			writeWords(hitObject, hitObjectAttribute);
		}

		void OpHitObjectGetClusterIdNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectGetClusterIdNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectGetCurrentTimeNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectGetCurrentTimeNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectGetGeometryIndexNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectGetGeometryIndexNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectGetHitKindNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectGetHitKindNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectGetInstanceCustomIndexNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectGetInstanceCustomIndexNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectGetInstanceIdNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectGetInstanceIdNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectGetLSSPositionsNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectGetLSSPositionsNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectGetLSSRadiiNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectGetLSSRadiiNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectGetObjectRayDirectionNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectGetObjectRayDirectionNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectGetObjectRayOriginNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectGetObjectRayOriginNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectGetObjectToWorldNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectGetObjectToWorldNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectGetPrimitiveIndexNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectGetPrimitiveIndexNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectGetRayTMaxNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectGetRayTMaxNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectGetRayTMinNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectGetRayTMinNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectGetShaderBindingTableRecordIndexNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectGetShaderBindingTableRecordIndexNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectGetShaderRecordBufferHandleNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectGetShaderRecordBufferHandleNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectGetSpherePositionNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectGetSpherePositionNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectGetSphereRadiusNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectGetSphereRadiusNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectGetWorldRayDirectionNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectGetWorldRayDirectionNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectGetWorldRayOriginNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectGetWorldRayOriginNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectGetWorldToObjectNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectGetWorldToObjectNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectIsEmptyNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectIsEmptyNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectIsHitNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectIsHitNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectIsLSSHitNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectIsLSSHitNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectIsMissNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectIsMissNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectIsSphereHitNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hitObject)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpHitObjectIsSphereHitNV, wordCount);
			writeWords(idResultType, idResult, hitObject);
		}

		void OpHitObjectRecordEmptyNV(IdRef hitObject)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpHitObjectRecordEmptyNV, wordCount);
			writeWords(hitObject);
		}

		void OpHitObjectRecordHitMotionNV(
			IdRef hitObject,
			IdRef accelerationStructure,
			IdRef instanceId,
			IdRef primitiveId,
			IdRef geometryIndex,
			IdRef hitKind,
			IdRef sBTRecordOffset,
			IdRef sBTRecordStride,
			IdRef origin,
			IdRef tMin,
			IdRef direction,
			IdRef tMax,
			IdRef currentTime,
			IdRef hitObjectAttributes)
		{
			uint16_t wordCount = 15;

			writeWord(spv::Op::OpHitObjectRecordHitMotionNV, wordCount);
			writeWords(hitObject, accelerationStructure, instanceId, primitiveId, geometryIndex, hitKind, sBTRecordOffset, sBTRecordStride, origin, tMin, direction, tMax, currentTime, hitObjectAttributes);
		}

		void OpHitObjectRecordHitNV(
			IdRef hitObject,
			IdRef accelerationStructure,
			IdRef instanceId,
			IdRef primitiveId,
			IdRef geometryIndex,
			IdRef hitKind,
			IdRef sBTRecordOffset,
			IdRef sBTRecordStride,
			IdRef origin,
			IdRef tMin,
			IdRef direction,
			IdRef tMax,
			IdRef hitObjectAttributes)
		{
			uint16_t wordCount = 14;

			writeWord(spv::Op::OpHitObjectRecordHitNV, wordCount);
			writeWords(hitObject, accelerationStructure, instanceId, primitiveId, geometryIndex, hitKind, sBTRecordOffset, sBTRecordStride, origin, tMin, direction, tMax, hitObjectAttributes);
		}

		void OpHitObjectRecordHitWithIndexMotionNV(
			IdRef hitObject,
			IdRef accelerationStructure,
			IdRef instanceId,
			IdRef primitiveId,
			IdRef geometryIndex,
			IdRef hitKind,
			IdRef sBTRecordIndex,
			IdRef origin,
			IdRef tMin,
			IdRef direction,
			IdRef tMax,
			IdRef currentTime,
			IdRef hitObjectAttributes)
		{
			uint16_t wordCount = 14;

			writeWord(spv::Op::OpHitObjectRecordHitWithIndexMotionNV, wordCount);
			writeWords(hitObject, accelerationStructure, instanceId, primitiveId, geometryIndex, hitKind, sBTRecordIndex, origin, tMin, direction, tMax, currentTime, hitObjectAttributes);
		}

		void OpHitObjectRecordHitWithIndexNV(
			IdRef hitObject,
			IdRef accelerationStructure,
			IdRef instanceId,
			IdRef primitiveId,
			IdRef geometryIndex,
			IdRef hitKind,
			IdRef sBTRecordIndex,
			IdRef origin,
			IdRef tMin,
			IdRef direction,
			IdRef tMax,
			IdRef hitObjectAttributes)
		{
			uint16_t wordCount = 13;

			writeWord(spv::Op::OpHitObjectRecordHitWithIndexNV, wordCount);
			writeWords(hitObject, accelerationStructure, instanceId, primitiveId, geometryIndex, hitKind, sBTRecordIndex, origin, tMin, direction, tMax, hitObjectAttributes);
		}

		void OpHitObjectRecordMissMotionNV(
			IdRef hitObject,
			IdRef sBTIndex,
			IdRef origin,
			IdRef tMin,
			IdRef direction,
			IdRef tMax,
			IdRef currentTime)
		{
			uint16_t wordCount = 8;

			writeWord(spv::Op::OpHitObjectRecordMissMotionNV, wordCount);
			writeWords(hitObject, sBTIndex, origin, tMin, direction, tMax, currentTime);
		}

		void OpHitObjectRecordMissNV(
			IdRef hitObject,
			IdRef sBTIndex,
			IdRef origin,
			IdRef tMin,
			IdRef direction,
			IdRef tMax)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpHitObjectRecordMissNV, wordCount);
			writeWords(hitObject, sBTIndex, origin, tMin, direction, tMax);
		}

		void OpHitObjectTraceRayMotionNV(
			IdRef hitObject,
			IdRef accelerationStructure,
			IdRef rayFlags,
			IdRef cullmask,
			IdRef sBTRecordOffset,
			IdRef sBTRecordStride,
			IdRef missIndex,
			IdRef origin,
			IdRef tMin,
			IdRef direction,
			IdRef tMax,
			IdRef time,
			IdRef payload)
		{
			uint16_t wordCount = 14;

			writeWord(spv::Op::OpHitObjectTraceRayMotionNV, wordCount);
			writeWords(hitObject, accelerationStructure, rayFlags, cullmask, sBTRecordOffset, sBTRecordStride, missIndex, origin, tMin, direction, tMax, time, payload);
		}

		void OpHitObjectTraceRayNV(
			IdRef hitObject,
			IdRef accelerationStructure,
			IdRef rayFlags,
			IdRef cullmask,
			IdRef sBTRecordOffset,
			IdRef sBTRecordStride,
			IdRef missIndex,
			IdRef origin,
			IdRef tMin,
			IdRef direction,
			IdRef tMax,
			IdRef payload)
		{
			uint16_t wordCount = 13;

			writeWord(spv::Op::OpHitObjectTraceRayNV, wordCount);
			writeWords(hitObject, accelerationStructure, rayFlags, cullmask, sBTRecordOffset, sBTRecordStride, missIndex, origin, tMin, direction, tMax, payload);
		}

		void OpIAdd(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpIAdd, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpIAddCarry(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpIAddCarry, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpIAddSatINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpIAddSatINTEL, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpIAverageINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpIAverageINTEL, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpIAverageRoundedINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpIAverageRoundedINTEL, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpIEqual(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpIEqual, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpIMul(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpIMul, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpIMul32x16INTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpIMul32x16INTEL, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpINotEqual(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpINotEqual, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpISub(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpISub, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpISubBorrow(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpISubBorrow, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpISubSatINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpISubSatINTEL, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpIgnoreIntersectionKHR()
		{
			uint16_t wordCount = 1;

			writeWord(spv::Op::OpIgnoreIntersectionKHR, wordCount);
			writeWords();
		}

		void OpIgnoreIntersectionNV()
		{
			uint16_t wordCount = 1;

			writeWord(spv::Op::OpIgnoreIntersectionNV, wordCount);
			writeWords();
		}

		void OpImage(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpImage, wordCount);
			writeWords(idResultType, idResult, sampledImage);
		}

		void OpImageBlockMatchGatherSADQCOM(
			IdResultType idResultType,
			IdResult idResult,
			IdRef targetSampledImage,
			IdRef targetCoordinates,
			IdRef referenceSampledImage,
			IdRef referenceCoordinates,
			IdRef blockSize)
		{
			uint16_t wordCount = 8;

			writeWord(spv::Op::OpImageBlockMatchGatherSADQCOM, wordCount);
			writeWords(idResultType, idResult, targetSampledImage, targetCoordinates, referenceSampledImage, referenceCoordinates, blockSize);
		}

		void OpImageBlockMatchGatherSSDQCOM(
			IdResultType idResultType,
			IdResult idResult,
			IdRef targetSampledImage,
			IdRef targetCoordinates,
			IdRef referenceSampledImage,
			IdRef referenceCoordinates,
			IdRef blockSize)
		{
			uint16_t wordCount = 8;

			writeWord(spv::Op::OpImageBlockMatchGatherSSDQCOM, wordCount);
			writeWords(idResultType, idResult, targetSampledImage, targetCoordinates, referenceSampledImage, referenceCoordinates, blockSize);
		}

		void OpImageBlockMatchSADQCOM(
			IdResultType idResultType,
			IdResult idResult,
			IdRef target,
			IdRef targetCoordinates,
			IdRef reference,
			IdRef referenceCoordinates,
			IdRef blockSize)
		{
			uint16_t wordCount = 8;

			writeWord(spv::Op::OpImageBlockMatchSADQCOM, wordCount);
			writeWords(idResultType, idResult, target, targetCoordinates, reference, referenceCoordinates, blockSize);
		}

		void OpImageBlockMatchSSDQCOM(
			IdResultType idResultType,
			IdResult idResult,
			IdRef target,
			IdRef targetCoordinates,
			IdRef reference,
			IdRef referenceCoordinates,
			IdRef blockSize)
		{
			uint16_t wordCount = 8;

			writeWord(spv::Op::OpImageBlockMatchSSDQCOM, wordCount);
			writeWords(idResultType, idResult, target, targetCoordinates, reference, referenceCoordinates, blockSize);
		}

		void OpImageBlockMatchWindowSADQCOM(
			IdResultType idResultType,
			IdResult idResult,
			IdRef targetSampledImage,
			IdRef targetCoordinates,
			IdRef referenceSampledImage,
			IdRef referenceCoordinates,
			IdRef blockSize)
		{
			uint16_t wordCount = 8;

			writeWord(spv::Op::OpImageBlockMatchWindowSADQCOM, wordCount);
			writeWords(idResultType, idResult, targetSampledImage, targetCoordinates, referenceSampledImage, referenceCoordinates, blockSize);
		}

		void OpImageBlockMatchWindowSSDQCOM(
			IdResultType idResultType,
			IdResult idResult,
			IdRef targetSampledImage,
			IdRef targetCoordinates,
			IdRef referenceSampledImage,
			IdRef referenceCoordinates,
			IdRef blockSize)
		{
			uint16_t wordCount = 8;

			writeWord(spv::Op::OpImageBlockMatchWindowSSDQCOM, wordCount);
			writeWords(idResultType, idResult, targetSampledImage, targetCoordinates, referenceSampledImage, referenceCoordinates, blockSize);
		}

		void OpImageBoxFilterQCOM(
			IdResultType idResultType,
			IdResult idResult,
			IdRef texture,
			IdRef coordinates,
			IdRef boxSize)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpImageBoxFilterQCOM, wordCount);
			writeWords(idResultType, idResult, texture, coordinates, boxSize);
		}

		void OpImageDrefGather(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate,
			IdRef dref,
			std::optional<spv::ImageOperandsMask> imageOperands = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, imageOperands);

			writeWord(spv::Op::OpImageDrefGather, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate, dref, imageOperands);
		}

		void OpImageFetch(
			IdResultType idResultType,
			IdResult idResult,
			IdRef image,
			IdRef coordinate,
			std::optional<spv::ImageOperandsMask> imageOperands = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, imageOperands);

			writeWord(spv::Op::OpImageFetch, wordCount);
			writeWords(idResultType, idResult, image, coordinate, imageOperands);
		}

		void OpImageGather(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate,
			IdRef component,
			std::optional<spv::ImageOperandsMask> imageOperands = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, imageOperands);

			writeWord(spv::Op::OpImageGather, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate, component, imageOperands);
		}

		void OpImageQueryFormat(
			IdResultType idResultType,
			IdResult idResult,
			IdRef image)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpImageQueryFormat, wordCount);
			writeWords(idResultType, idResult, image);
		}

		void OpImageQueryLevels(
			IdResultType idResultType,
			IdResult idResult,
			IdRef image)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpImageQueryLevels, wordCount);
			writeWords(idResultType, idResult, image);
		}

		void OpImageQueryLod(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpImageQueryLod, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate);
		}

		void OpImageQueryOrder(
			IdResultType idResultType,
			IdResult idResult,
			IdRef image)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpImageQueryOrder, wordCount);
			writeWords(idResultType, idResult, image);
		}

		void OpImageQuerySamples(
			IdResultType idResultType,
			IdResult idResult,
			IdRef image)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpImageQuerySamples, wordCount);
			writeWords(idResultType, idResult, image);
		}

		void OpImageQuerySize(
			IdResultType idResultType,
			IdResult idResult,
			IdRef image)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpImageQuerySize, wordCount);
			writeWords(idResultType, idResult, image);
		}

		void OpImageQuerySizeLod(
			IdResultType idResultType,
			IdResult idResult,
			IdRef image,
			IdRef levelOfDetail)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpImageQuerySizeLod, wordCount);
			writeWords(idResultType, idResult, image, levelOfDetail);
		}

		void OpImageRead(
			IdResultType idResultType,
			IdResult idResult,
			IdRef image,
			IdRef coordinate,
			std::optional<spv::ImageOperandsMask> imageOperands = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, imageOperands);

			writeWord(spv::Op::OpImageRead, wordCount);
			writeWords(idResultType, idResult, image, coordinate, imageOperands);
		}

		void OpImageSampleDrefExplicitLod(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate,
			IdRef dref,
			spv::ImageOperandsMask imageOperands)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpImageSampleDrefExplicitLod, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate, dref, imageOperands);
		}

		void OpImageSampleDrefImplicitLod(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate,
			IdRef dref,
			std::optional<spv::ImageOperandsMask> imageOperands = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, imageOperands);

			writeWord(spv::Op::OpImageSampleDrefImplicitLod, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate, dref, imageOperands);
		}

		void OpImageSampleExplicitLod(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate,
			spv::ImageOperandsMask imageOperands)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpImageSampleExplicitLod, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate, imageOperands);
		}

		void OpImageSampleFootprintNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate,
			IdRef granularity,
			IdRef coarse,
			std::optional<spv::ImageOperandsMask> imageOperands = {})
		{
			uint16_t wordCount = 7;
			countOperandsWord(wordCount, imageOperands);

			writeWord(spv::Op::OpImageSampleFootprintNV, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate, granularity, coarse, imageOperands);
		}

		void OpImageSampleImplicitLod(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate,
			std::optional<spv::ImageOperandsMask> imageOperands = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, imageOperands);

			writeWord(spv::Op::OpImageSampleImplicitLod, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate, imageOperands);
		}

		void OpImageSampleProjDrefExplicitLod(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate,
			IdRef dref,
			spv::ImageOperandsMask imageOperands)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpImageSampleProjDrefExplicitLod, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate, dref, imageOperands);
		}

		void OpImageSampleProjDrefImplicitLod(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate,
			IdRef dref,
			std::optional<spv::ImageOperandsMask> imageOperands = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, imageOperands);

			writeWord(spv::Op::OpImageSampleProjDrefImplicitLod, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate, dref, imageOperands);
		}

		void OpImageSampleProjExplicitLod(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate,
			spv::ImageOperandsMask imageOperands)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpImageSampleProjExplicitLod, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate, imageOperands);
		}

		void OpImageSampleProjImplicitLod(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate,
			std::optional<spv::ImageOperandsMask> imageOperands = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, imageOperands);

			writeWord(spv::Op::OpImageSampleProjImplicitLod, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate, imageOperands);
		}

		void OpImageSampleWeightedQCOM(
			IdResultType idResultType,
			IdResult idResult,
			IdRef texture,
			IdRef coordinates,
			IdRef weights)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpImageSampleWeightedQCOM, wordCount);
			writeWords(idResultType, idResult, texture, coordinates, weights);
		}

		void OpImageSparseDrefGather(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate,
			IdRef dref,
			std::optional<spv::ImageOperandsMask> imageOperands = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, imageOperands);

			writeWord(spv::Op::OpImageSparseDrefGather, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate, dref, imageOperands);
		}

		void OpImageSparseFetch(
			IdResultType idResultType,
			IdResult idResult,
			IdRef image,
			IdRef coordinate,
			std::optional<spv::ImageOperandsMask> imageOperands = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, imageOperands);

			writeWord(spv::Op::OpImageSparseFetch, wordCount);
			writeWords(idResultType, idResult, image, coordinate, imageOperands);
		}

		void OpImageSparseGather(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate,
			IdRef component,
			std::optional<spv::ImageOperandsMask> imageOperands = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, imageOperands);

			writeWord(spv::Op::OpImageSparseGather, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate, component, imageOperands);
		}

		void OpImageSparseRead(
			IdResultType idResultType,
			IdResult idResult,
			IdRef image,
			IdRef coordinate,
			std::optional<spv::ImageOperandsMask> imageOperands = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, imageOperands);

			writeWord(spv::Op::OpImageSparseRead, wordCount);
			writeWords(idResultType, idResult, image, coordinate, imageOperands);
		}

		void OpImageSparseSampleDrefExplicitLod(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate,
			IdRef dref,
			spv::ImageOperandsMask imageOperands)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpImageSparseSampleDrefExplicitLod, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate, dref, imageOperands);
		}

		void OpImageSparseSampleDrefImplicitLod(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate,
			IdRef dref,
			std::optional<spv::ImageOperandsMask> imageOperands = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, imageOperands);

			writeWord(spv::Op::OpImageSparseSampleDrefImplicitLod, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate, dref, imageOperands);
		}

		void OpImageSparseSampleExplicitLod(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate,
			spv::ImageOperandsMask imageOperands)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpImageSparseSampleExplicitLod, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate, imageOperands);
		}

		void OpImageSparseSampleImplicitLod(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate,
			std::optional<spv::ImageOperandsMask> imageOperands = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, imageOperands);

			writeWord(spv::Op::OpImageSparseSampleImplicitLod, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate, imageOperands);
		}

		void OpImageSparseSampleProjDrefExplicitLod(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate,
			IdRef dref,
			spv::ImageOperandsMask imageOperands)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpImageSparseSampleProjDrefExplicitLod, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate, dref, imageOperands);
		}

		void OpImageSparseSampleProjDrefImplicitLod(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate,
			IdRef dref,
			std::optional<spv::ImageOperandsMask> imageOperands = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, imageOperands);

			writeWord(spv::Op::OpImageSparseSampleProjDrefImplicitLod, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate, dref, imageOperands);
		}

		void OpImageSparseSampleProjExplicitLod(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate,
			spv::ImageOperandsMask imageOperands)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpImageSparseSampleProjExplicitLod, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate, imageOperands);
		}

		void OpImageSparseSampleProjImplicitLod(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sampledImage,
			IdRef coordinate,
			std::optional<spv::ImageOperandsMask> imageOperands = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, imageOperands);

			writeWord(spv::Op::OpImageSparseSampleProjImplicitLod, wordCount);
			writeWords(idResultType, idResult, sampledImage, coordinate, imageOperands);
		}

		void OpImageSparseTexelsResident(
			IdResultType idResultType,
			IdResult idResult,
			IdRef residentCode)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpImageSparseTexelsResident, wordCount);
			writeWords(idResultType, idResult, residentCode);
		}

		void OpImageTexelPointer(
			IdResultType idResultType,
			IdResult idResult,
			IdRef image,
			IdRef coordinate,
			IdRef sample)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpImageTexelPointer, wordCount);
			writeWords(idResultType, idResult, image, coordinate, sample);
		}

		void OpImageWrite(
			IdRef image,
			IdRef coordinate,
			IdRef texel,
			std::optional<spv::ImageOperandsMask> imageOperands = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, imageOperands);

			writeWord(spv::Op::OpImageWrite, wordCount);
			writeWords(image, coordinate, texel, imageOperands);
		}

		void OpInBoundsAccessChain(
			IdResultType idResultType,
			IdResult idResult,
			IdRef base,
			const std::vector<IdRef>& indexes = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, indexes);

			writeWord(spv::Op::OpInBoundsAccessChain, wordCount);
			writeWords(idResultType, idResult, base, indexes);
		}

		void OpInBoundsPtrAccessChain(
			IdResultType idResultType,
			IdResult idResult,
			IdRef base,
			IdRef element,
			const std::vector<IdRef>& indexes = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, indexes);

			writeWord(spv::Op::OpInBoundsPtrAccessChain, wordCount);
			writeWords(idResultType, idResult, base, element, indexes);
		}

		void OpIsFinite(
			IdResultType idResultType,
			IdResult idResult,
			IdRef x)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpIsFinite, wordCount);
			writeWords(idResultType, idResult, x);
		}

		void OpIsHelperInvocationEXT(
			IdResultType idResultType,
			IdResult idResult)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpIsHelperInvocationEXT, wordCount);
			writeWords(idResultType, idResult);
		}

		void OpIsInf(
			IdResultType idResultType,
			IdResult idResult,
			IdRef x)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpIsInf, wordCount);
			writeWords(idResultType, idResult, x);
		}

		void OpIsNan(
			IdResultType idResultType,
			IdResult idResult,
			IdRef x)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpIsNan, wordCount);
			writeWords(idResultType, idResult, x);
		}

		void OpIsNodePayloadValidAMDX(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payloadType,
			IdRef nodeIndex)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpIsNodePayloadValidAMDX, wordCount);
			writeWords(idResultType, idResult, payloadType, nodeIndex);
		}

		void OpIsNormal(
			IdResultType idResultType,
			IdResult idResult,
			IdRef x)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpIsNormal, wordCount);
			writeWords(idResultType, idResult, x);
		}

		void OpIsValidEvent(
			IdResultType idResultType,
			IdResult idResult,
			IdRef event)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpIsValidEvent, wordCount);
			writeWords(idResultType, idResult, event);
		}

		void OpIsValidReserveId(
			IdResultType idResultType,
			IdResult idResult,
			IdRef reserveId)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpIsValidReserveId, wordCount);
			writeWords(idResultType, idResult, reserveId);
		}

		void OpKill()
		{
			uint16_t wordCount = 1;

			writeWord(spv::Op::OpKill, wordCount);
			writeWords();
		}

		void OpLabel(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpLabel, wordCount);
			writeWords(idResult);
		}

		void OpLessOrGreater(
			IdResultType idResultType,
			IdResult idResult,
			IdRef x,
			IdRef y)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpLessOrGreater, wordCount);
			writeWords(idResultType, idResult, x, y);
		}

		void OpLifetimeStart(
			IdRef pointer,
			uint32_t size)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpLifetimeStart, wordCount);
			writeWords(pointer, size);
		}

		void OpLifetimeStop(
			IdRef pointer,
			uint32_t size)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpLifetimeStop, wordCount);
			writeWords(pointer, size);
		}

		void OpLine(
			IdRef file,
			uint32_t line,
			uint32_t column)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpLine, wordCount);
			writeWords(file, line, column);
		}

		void OpLoad(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer,
			std::optional<spv::MemoryAccessMask> memoryAccess = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, memoryAccess);

			writeWord(spv::Op::OpLoad, wordCount);
			writeWords(idResultType, idResult, pointer, memoryAccess);
		}

		void OpLogicalAnd(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpLogicalAnd, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpLogicalEqual(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpLogicalEqual, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpLogicalNot(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpLogicalNot, wordCount);
			writeWords(idResultType, idResult, operand);
		}

		void OpLogicalNotEqual(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpLogicalNotEqual, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpLogicalOr(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpLogicalOr, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpLoopControlINTEL(const std::vector<uint32_t>& loopControlParameters = {})
		{
			uint16_t wordCount = 1;
			countOperandsWord(wordCount, loopControlParameters);

			writeWord(spv::Op::OpLoopControlINTEL, wordCount);
			writeWords(loopControlParameters);
		}

		void OpLoopMerge(
			IdRef mergeBlock,
			IdRef continueTarget,
			spv::LoopControlMask loopControl)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpLoopMerge, wordCount);
			writeWords(mergeBlock, continueTarget, loopControl);
		}

		void OpMaskedGatherINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef ptrVector,
			uint32_t alignment,
			IdRef mask,
			IdRef fillEmpty)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpMaskedGatherINTEL, wordCount);
			writeWords(idResultType, idResult, ptrVector, alignment, mask, fillEmpty);
		}

		void OpMaskedScatterINTEL(
			IdRef inputVector,
			IdRef ptrVector,
			uint32_t alignment,
			IdRef mask)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpMaskedScatterINTEL, wordCount);
			writeWords(inputVector, ptrVector, alignment, mask);
		}

		void OpMatrixTimesMatrix(
			IdResultType idResultType,
			IdResult idResult,
			IdRef leftMatrix,
			IdRef rightMatrix)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpMatrixTimesMatrix, wordCount);
			writeWords(idResultType, idResult, leftMatrix, rightMatrix);
		}

		void OpMatrixTimesScalar(
			IdResultType idResultType,
			IdResult idResult,
			IdRef matrix,
			IdRef scalar)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpMatrixTimesScalar, wordCount);
			writeWords(idResultType, idResult, matrix, scalar);
		}

		void OpMatrixTimesVector(
			IdResultType idResultType,
			IdResult idResult,
			IdRef matrix,
			IdRef vector)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpMatrixTimesVector, wordCount);
			writeWords(idResultType, idResult, matrix, vector);
		}

		void OpMemberDecorate(
			IdRef structureType,
			uint32_t member,
			spv::Decoration decoration)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpMemberDecorate, wordCount);
			writeWords(structureType, member, decoration);
		}

		void OpMemberDecorateString(
			IdRef structType,
			uint32_t member,
			spv::Decoration decoration)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpMemberDecorateString, wordCount);
			writeWords(structType, member, decoration);
		}

		void OpMemberName(
			IdRef type,
			uint32_t member,
			const std::string& name)
		{
			uint16_t wordCount = 3;
			countOperandsWord(wordCount, name);

			writeWord(spv::Op::OpMemberName, wordCount);
			writeWords(type, member, name);
		}

		void OpMemoryBarrier(
			IdScope memory,
			IdMemorySemantics semantics)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpMemoryBarrier, wordCount);
			writeWords(memory, semantics);
		}

		void OpMemoryModel(
			spv::AddressingModel addressingModel,
			spv::MemoryModel memoryModel)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpMemoryModel, wordCount);
			writeWords(addressingModel, memoryModel);
		}

		void OpMemoryNamedBarrier(
			IdRef namedBarrier,
			IdScope memory,
			IdMemorySemantics semantics)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpMemoryNamedBarrier, wordCount);
			writeWords(namedBarrier, memory, semantics);
		}

		void OpModuleProcessed(const std::string& process)
		{
			uint16_t wordCount = 1;
			countOperandsWord(wordCount, process);

			writeWord(spv::Op::OpModuleProcessed, wordCount);
			writeWords(process);
		}

		void OpName(
			IdRef target,
			const std::string& name)
		{
			uint16_t wordCount = 2;
			countOperandsWord(wordCount, name);

			writeWord(spv::Op::OpName, wordCount);
			writeWords(target, name);
		}

		void OpNamedBarrierInitialize(
			IdResultType idResultType,
			IdResult idResult,
			IdRef subgroupCount)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpNamedBarrierInitialize, wordCount);
			writeWords(idResultType, idResult, subgroupCount);
		}

		void OpNoLine()
		{
			uint16_t wordCount = 1;

			writeWord(spv::Op::OpNoLine, wordCount);
			writeWords();
		}

		void OpNodePayloadArrayLengthAMDX(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payloadArray)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpNodePayloadArrayLengthAMDX, wordCount);
			writeWords(idResultType, idResult, payloadArray);
		}

		void OpNop()
		{
			uint16_t wordCount = 1;

			writeWord(spv::Op::OpNop, wordCount);
			writeWords();
		}

		void OpNot(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpNot, wordCount);
			writeWords(idResultType, idResult, operand);
		}

		void OpOrdered(
			IdResultType idResultType,
			IdResult idResult,
			IdRef x,
			IdRef y)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpOrdered, wordCount);
			writeWords(idResultType, idResult, x, y);
		}

		void OpOuterProduct(
			IdResultType idResultType,
			IdResult idResult,
			IdRef vector1,
			IdRef vector2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpOuterProduct, wordCount);
			writeWords(idResultType, idResult, vector1, vector2);
		}

		void OpPhi(
			IdResultType idResultType,
			IdResult idResult,
			const std::vector<std::tuple<IdRef, IdRef>>& variableParents = {})
		{
			uint16_t wordCount = 3;
			countOperandsWord(wordCount, variableParents);

			writeWord(spv::Op::OpPhi, wordCount);
			writeWords(idResultType, idResult, variableParents);
		}

		void OpPtrAccessChain(
			IdResultType idResultType,
			IdResult idResult,
			IdRef base,
			IdRef element,
			const std::vector<IdRef>& indexes = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, indexes);

			writeWord(spv::Op::OpPtrAccessChain, wordCount);
			writeWords(idResultType, idResult, base, element, indexes);
		}

		void OpPtrCastToCrossWorkgroupINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpPtrCastToCrossWorkgroupINTEL, wordCount);
			writeWords(idResultType, idResult, pointer);
		}

		void OpPtrCastToGeneric(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpPtrCastToGeneric, wordCount);
			writeWords(idResultType, idResult, pointer);
		}

		void OpPtrDiff(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpPtrDiff, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpPtrEqual(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpPtrEqual, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpPtrNotEqual(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpPtrNotEqual, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpQuantizeToF16(
			IdResultType idResultType,
			IdResult idResult,
			IdRef value)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpQuantizeToF16, wordCount);
			writeWords(idResultType, idResult, value);
		}

		void OpRawAccessChainNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef base,
			IdRef byteStride,
			IdRef elementIndex,
			IdRef byteOffset,
			std::optional<spv::RawAccessChainOperandsMask> rawAccessChainOperands = {})
		{
			uint16_t wordCount = 7;
			countOperandsWord(wordCount, rawAccessChainOperands);

			writeWord(spv::Op::OpRawAccessChainNV, wordCount);
			writeWords(idResultType, idResult, base, byteStride, elementIndex, byteOffset, rawAccessChainOperands);
		}

		void OpRayQueryConfirmIntersectionKHR(IdRef rayQuery)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpRayQueryConfirmIntersectionKHR, wordCount);
			writeWords(rayQuery);
		}

		void OpRayQueryGenerateIntersectionKHR(
			IdRef rayQuery,
			IdRef hitT)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpRayQueryGenerateIntersectionKHR, wordCount);
			writeWords(rayQuery, hitT);
		}

		void OpRayQueryGetClusterIdNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryGetClusterIdNV, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryGetIntersectionBarycentricsKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryGetIntersectionBarycentricsKHR, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryGetIntersectionCandidateAABBOpaqueKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpRayQueryGetIntersectionCandidateAABBOpaqueKHR, wordCount);
			writeWords(idResultType, idResult, rayQuery);
		}

		void OpRayQueryGetIntersectionFrontFaceKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryGetIntersectionFrontFaceKHR, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryGetIntersectionGeometryIndexKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryGetIntersectionGeometryIndexKHR, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryGetIntersectionInstanceCustomIndexKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryGetIntersectionInstanceCustomIndexKHR, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryGetIntersectionInstanceIdKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryGetIntersectionInstanceIdKHR, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryGetIntersectionLSSHitValueNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryGetIntersectionLSSHitValueNV, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryGetIntersectionLSSPositionsNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryGetIntersectionLSSPositionsNV, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryGetIntersectionLSSRadiiNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryGetIntersectionLSSRadiiNV, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryGetIntersectionObjectRayDirectionKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryGetIntersectionObjectRayDirectionKHR, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryGetIntersectionObjectRayOriginKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryGetIntersectionObjectRayOriginKHR, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryGetIntersectionObjectToWorldKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryGetIntersectionObjectToWorldKHR, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryGetIntersectionPrimitiveIndexKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryGetIntersectionPrimitiveIndexKHR, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryGetIntersectionSpherePositionNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryGetIntersectionSpherePositionNV, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryGetIntersectionSphereRadiusNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryGetIntersectionSphereRadiusNV, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryGetIntersectionTKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryGetIntersectionTKHR, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryGetIntersectionTriangleVertexPositionsKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryGetIntersectionTriangleVertexPositionsKHR, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryGetIntersectionTypeKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryGetIntersectionTypeKHR, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryGetIntersectionWorldToObjectKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryGetIntersectionWorldToObjectKHR, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryGetRayFlagsKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpRayQueryGetRayFlagsKHR, wordCount);
			writeWords(idResultType, idResult, rayQuery);
		}

		void OpRayQueryGetRayTMinKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpRayQueryGetRayTMinKHR, wordCount);
			writeWords(idResultType, idResult, rayQuery);
		}

		void OpRayQueryGetWorldRayDirectionKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpRayQueryGetWorldRayDirectionKHR, wordCount);
			writeWords(idResultType, idResult, rayQuery);
		}

		void OpRayQueryGetWorldRayOriginKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpRayQueryGetWorldRayOriginKHR, wordCount);
			writeWords(idResultType, idResult, rayQuery);
		}

		void OpRayQueryInitializeKHR(
			IdRef rayQuery,
			IdRef accel,
			IdRef rayFlags,
			IdRef cullMask,
			IdRef rayOrigin,
			IdRef rayTMin,
			IdRef rayDirection,
			IdRef rayTMax)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpRayQueryInitializeKHR, wordCount);
			writeWords(rayQuery, accel, rayFlags, cullMask, rayOrigin, rayTMin, rayDirection, rayTMax);
		}

		void OpRayQueryIsLSSHitNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryIsLSSHitNV, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryIsSphereHitNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery,
			IdRef intersection)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpRayQueryIsSphereHitNV, wordCount);
			writeWords(idResultType, idResult, rayQuery, intersection);
		}

		void OpRayQueryProceedKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef rayQuery)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpRayQueryProceedKHR, wordCount);
			writeWords(idResultType, idResult, rayQuery);
		}

		void OpRayQueryTerminateKHR(IdRef rayQuery)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpRayQueryTerminateKHR, wordCount);
			writeWords(rayQuery);
		}

		void OpReadClockKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdScope scope)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpReadClockKHR, wordCount);
			writeWords(idResultType, idResult, scope);
		}

		void OpReadPipe(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pipe,
			IdRef pointer,
			IdRef packetSize,
			IdRef packetAlignment)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpReadPipe, wordCount);
			writeWords(idResultType, idResult, pipe, pointer, packetSize, packetAlignment);
		}

		void OpReadPipeBlockingINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef packetSize,
			IdRef packetAlignment)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpReadPipeBlockingINTEL, wordCount);
			writeWords(idResultType, idResult, packetSize, packetAlignment);
		}

		void OpReleaseEvent(IdRef event)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpReleaseEvent, wordCount);
			writeWords(event);
		}

		void OpReorderThreadWithHintNV(
			IdRef hint,
			IdRef bits)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpReorderThreadWithHintNV, wordCount);
			writeWords(hint, bits);
		}

		void OpReorderThreadWithHitObjectNV(
			IdRef hitObject,
			std::optional<IdRef> hint = {},
			std::optional<IdRef> bits = {})
		{
			uint16_t wordCount = 2;
			countOperandsWord(wordCount, hint, bits);

			writeWord(spv::Op::OpReorderThreadWithHitObjectNV, wordCount);
			writeWords(hitObject, hint, bits);
		}

		void OpReportIntersectionKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef hit,
			IdRef hitKind)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpReportIntersectionKHR, wordCount);
			writeWords(idResultType, idResult, hit, hitKind);
		}

		void OpReserveReadPipePackets(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pipe,
			IdRef numPackets,
			IdRef packetSize,
			IdRef packetAlignment)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpReserveReadPipePackets, wordCount);
			writeWords(idResultType, idResult, pipe, numPackets, packetSize, packetAlignment);
		}

		void OpReserveWritePipePackets(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pipe,
			IdRef numPackets,
			IdRef packetSize,
			IdRef packetAlignment)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpReserveWritePipePackets, wordCount);
			writeWords(idResultType, idResult, pipe, numPackets, packetSize, packetAlignment);
		}

		void OpReservedReadPipe(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pipe,
			IdRef reserveId,
			IdRef index,
			IdRef pointer,
			IdRef packetSize,
			IdRef packetAlignment)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpReservedReadPipe, wordCount);
			writeWords(idResultType, idResult, pipe, reserveId, index, pointer, packetSize, packetAlignment);
		}

		void OpReservedWritePipe(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pipe,
			IdRef reserveId,
			IdRef index,
			IdRef pointer,
			IdRef packetSize,
			IdRef packetAlignment)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpReservedWritePipe, wordCount);
			writeWords(idResultType, idResult, pipe, reserveId, index, pointer, packetSize, packetAlignment);
		}

		void OpRestoreMemoryINTEL(IdRef ptr)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpRestoreMemoryINTEL, wordCount);
			writeWords(ptr);
		}

		void OpRetainEvent(IdRef event)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpRetainEvent, wordCount);
			writeWords(event);
		}

		void OpReturn()
		{
			uint16_t wordCount = 1;

			writeWord(spv::Op::OpReturn, wordCount);
			writeWords();
		}

		void OpReturnValue(IdRef value)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpReturnValue, wordCount);
			writeWords(value);
		}

		void OpRoundFToTF32INTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef floatValue)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpRoundFToTF32INTEL, wordCount);
			writeWords(idResultType, idResult, floatValue);
		}

		void OpSConvert(
			IdResultType idResultType,
			IdResult idResult,
			IdRef signedValue)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSConvert, wordCount);
			writeWords(idResultType, idResult, signedValue);
		}

		void OpSDiv(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSDiv, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpSDot(
			IdResultType idResultType,
			IdResult idResult,
			IdRef vector1,
			IdRef vector2,
			std::optional<spv::PackedVectorFormat> packedVectorFormat = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, packedVectorFormat);

			writeWord(spv::Op::OpSDot, wordCount);
			writeWords(idResultType, idResult, vector1, vector2, packedVectorFormat);
		}

		void OpSDotAccSat(
			IdResultType idResultType,
			IdResult idResult,
			IdRef vector1,
			IdRef vector2,
			IdRef accumulator,
			std::optional<spv::PackedVectorFormat> packedVectorFormat = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, packedVectorFormat);

			writeWord(spv::Op::OpSDotAccSat, wordCount);
			writeWords(idResultType, idResult, vector1, vector2, accumulator, packedVectorFormat);
		}

		void OpSGreaterThan(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSGreaterThan, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpSGreaterThanEqual(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSGreaterThanEqual, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpSLessThan(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSLessThan, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpSLessThanEqual(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSLessThanEqual, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpSMod(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSMod, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpSMulExtended(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSMulExtended, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpSNegate(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSNegate, wordCount);
			writeWords(idResultType, idResult, operand);
		}

		void OpSRem(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSRem, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpSUDot(
			IdResultType idResultType,
			IdResult idResult,
			IdRef vector1,
			IdRef vector2,
			std::optional<spv::PackedVectorFormat> packedVectorFormat = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, packedVectorFormat);

			writeWord(spv::Op::OpSUDot, wordCount);
			writeWords(idResultType, idResult, vector1, vector2, packedVectorFormat);
		}

		void OpSUDotAccSat(
			IdResultType idResultType,
			IdResult idResult,
			IdRef vector1,
			IdRef vector2,
			IdRef accumulator,
			std::optional<spv::PackedVectorFormat> packedVectorFormat = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, packedVectorFormat);

			writeWord(spv::Op::OpSUDotAccSat, wordCount);
			writeWords(idResultType, idResult, vector1, vector2, accumulator, packedVectorFormat);
		}

		void OpSampledImage(
			IdResultType idResultType,
			IdResult idResult,
			IdRef image,
			IdRef sampler)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSampledImage, wordCount);
			writeWords(idResultType, idResult, image, sampler);
		}

		void OpSamplerImageAddressingModeNV(uint32_t bitWidth)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpSamplerImageAddressingModeNV, wordCount);
			writeWords(bitWidth);
		}

		void OpSatConvertSToU(
			IdResultType idResultType,
			IdResult idResult,
			IdRef signedValue)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSatConvertSToU, wordCount);
			writeWords(idResultType, idResult, signedValue);
		}

		void OpSatConvertUToS(
			IdResultType idResultType,
			IdResult idResult,
			IdRef unsignedValue)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSatConvertUToS, wordCount);
			writeWords(idResultType, idResult, unsignedValue);
		}

		void OpSaveMemoryINTEL(
			IdResultType idResultType,
			IdResult idResult)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpSaveMemoryINTEL, wordCount);
			writeWords(idResultType, idResult);
		}

		void OpSelect(
			IdResultType idResultType,
			IdResult idResult,
			IdRef condition,
			IdRef object1,
			IdRef object2)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpSelect, wordCount);
			writeWords(idResultType, idResult, condition, object1, object2);
		}

		void OpSelectionMerge(
			IdRef mergeBlock,
			spv::SelectionControlMask selectionControl)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpSelectionMerge, wordCount);
			writeWords(mergeBlock, selectionControl);
		}

		void OpSetMeshOutputsEXT(
			IdRef vertexCount,
			IdRef primitiveCount)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpSetMeshOutputsEXT, wordCount);
			writeWords(vertexCount, primitiveCount);
		}

		void OpSetUserEventStatus(
			IdRef event,
			IdRef status)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpSetUserEventStatus, wordCount);
			writeWords(event, status);
		}

		void OpShiftLeftLogical(
			IdResultType idResultType,
			IdResult idResult,
			IdRef base,
			IdRef shift)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpShiftLeftLogical, wordCount);
			writeWords(idResultType, idResult, base, shift);
		}

		void OpShiftRightArithmetic(
			IdResultType idResultType,
			IdResult idResult,
			IdRef base,
			IdRef shift)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpShiftRightArithmetic, wordCount);
			writeWords(idResultType, idResult, base, shift);
		}

		void OpShiftRightLogical(
			IdResultType idResultType,
			IdResult idResult,
			IdRef base,
			IdRef shift)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpShiftRightLogical, wordCount);
			writeWords(idResultType, idResult, base, shift);
		}

		void OpSignBitSet(
			IdResultType idResultType,
			IdResult idResult,
			IdRef x)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSignBitSet, wordCount);
			writeWords(idResultType, idResult, x);
		}

		void OpSizeOf(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pointer)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSizeOf, wordCount);
			writeWords(idResultType, idResult, pointer);
		}

		void OpSource(
			spv::SourceLanguage sourceLanguage,
			uint32_t version,
			std::optional<IdRef> file = {},
			std::optional<std::string> source = {})
		{
			uint16_t wordCount = 3;
			countOperandsWord(wordCount, file, source);

			writeWord(spv::Op::OpSource, wordCount);
			writeWords(sourceLanguage, version, file, source);
		}

		void OpSourceContinued(const std::string& continuedSource)
		{
			uint16_t wordCount = 1;
			countOperandsWord(wordCount, continuedSource);

			writeWord(spv::Op::OpSourceContinued, wordCount);
			writeWords(continuedSource);
		}

		void OpSourceExtension(const std::string& extension)
		{
			uint16_t wordCount = 1;
			countOperandsWord(wordCount, extension);

			writeWord(spv::Op::OpSourceExtension, wordCount);
			writeWords(extension);
		}

		void OpSpecConstant(
			IdResultType idResultType,
			IdResult idResult,
			spvConstant auto value)
		{
			uint16_t wordCount = 3;
			countOperandsWord(wordCount, value);

			writeWord(spv::Op::OpSpecConstant, wordCount);
			writeWords(idResultType, idResult, value);
		}

		void OpSpecConstantComposite(
			IdResultType idResultType,
			IdResult idResult,
			const std::vector<IdRef>& constituents = {})
		{
			uint16_t wordCount = 3;
			countOperandsWord(wordCount, constituents);

			writeWord(spv::Op::OpSpecConstantComposite, wordCount);
			writeWords(idResultType, idResult, constituents);
		}

		void OpSpecConstantCompositeContinuedINTEL(const std::vector<IdRef>& constituents = {})
		{
			uint16_t wordCount = 1;
			countOperandsWord(wordCount, constituents);

			writeWord(spv::Op::OpSpecConstantCompositeContinuedINTEL, wordCount);
			writeWords(constituents);
		}

		void OpSpecConstantCompositeReplicateEXT(
			IdResultType idResultType,
			IdResult idResult,
			IdRef value)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSpecConstantCompositeReplicateEXT, wordCount);
			writeWords(idResultType, idResult, value);
		}

		void OpSpecConstantFalse(
			IdResultType idResultType,
			IdResult idResult)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpSpecConstantFalse, wordCount);
			writeWords(idResultType, idResult);
		}

		void OpSpecConstantOp(
			IdResultType idResultType,
			IdResult idResult,
			uint32_t opcode)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSpecConstantOp, wordCount);
			writeWords(idResultType, idResult, opcode);
		}

		void OpSpecConstantStringAMDX(
			IdResult idResult,
			const std::string& literalString)
		{
			uint16_t wordCount = 2;
			countOperandsWord(wordCount, literalString);

			writeWord(spv::Op::OpSpecConstantStringAMDX, wordCount);
			writeWords(idResult, literalString);
		}

		void OpSpecConstantTrue(
			IdResultType idResultType,
			IdResult idResult)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpSpecConstantTrue, wordCount);
			writeWords(idResultType, idResult);
		}

		void OpStencilAttachmentReadEXT(
			IdResultType idResultType,
			IdResult idResult,
			std::optional<IdRef> sample = {})
		{
			uint16_t wordCount = 3;
			countOperandsWord(wordCount, sample);

			writeWord(spv::Op::OpStencilAttachmentReadEXT, wordCount);
			writeWords(idResultType, idResult, sample);
		}

		void OpStore(
			IdRef pointer,
			IdRef object,
			std::optional<spv::MemoryAccessMask> memoryAccess = {})
		{
			uint16_t wordCount = 3;
			countOperandsWord(wordCount, memoryAccess);

			writeWord(spv::Op::OpStore, wordCount);
			writeWords(pointer, object, memoryAccess);
		}

		void OpString(
			IdResult idResult,
			const std::string& string)
		{
			uint16_t wordCount = 2;
			countOperandsWord(wordCount, string);

			writeWord(spv::Op::OpString, wordCount);
			writeWords(idResult, string);
		}

		void OpSubgroup2DBlockLoadINTEL(
			IdRef elementSize,
			IdRef blockWidth,
			IdRef blockHeight,
			IdRef blockCount,
			IdRef srcBasePointer,
			IdRef memoryWidth,
			IdRef memoryHeight,
			IdRef memoryPitch,
			IdRef coordinate,
			IdRef dstPointer)
		{
			uint16_t wordCount = 11;

			writeWord(spv::Op::OpSubgroup2DBlockLoadINTEL, wordCount);
			writeWords(elementSize, blockWidth, blockHeight, blockCount, srcBasePointer, memoryWidth, memoryHeight, memoryPitch, coordinate, dstPointer);
		}

		void OpSubgroup2DBlockLoadTransformINTEL(
			IdRef elementSize,
			IdRef blockWidth,
			IdRef blockHeight,
			IdRef blockCount,
			IdRef srcBasePointer,
			IdRef memoryWidth,
			IdRef memoryHeight,
			IdRef memoryPitch,
			IdRef coordinate,
			IdRef dstPointer)
		{
			uint16_t wordCount = 11;

			writeWord(spv::Op::OpSubgroup2DBlockLoadTransformINTEL, wordCount);
			writeWords(elementSize, blockWidth, blockHeight, blockCount, srcBasePointer, memoryWidth, memoryHeight, memoryPitch, coordinate, dstPointer);
		}

		void OpSubgroup2DBlockLoadTransposeINTEL(
			IdRef elementSize,
			IdRef blockWidth,
			IdRef blockHeight,
			IdRef blockCount,
			IdRef srcBasePointer,
			IdRef memoryWidth,
			IdRef memoryHeight,
			IdRef memoryPitch,
			IdRef coordinate,
			IdRef dstPointer)
		{
			uint16_t wordCount = 11;

			writeWord(spv::Op::OpSubgroup2DBlockLoadTransposeINTEL, wordCount);
			writeWords(elementSize, blockWidth, blockHeight, blockCount, srcBasePointer, memoryWidth, memoryHeight, memoryPitch, coordinate, dstPointer);
		}

		void OpSubgroup2DBlockPrefetchINTEL(
			IdRef elementSize,
			IdRef blockWidth,
			IdRef blockHeight,
			IdRef blockCount,
			IdRef srcBasePointer,
			IdRef memoryWidth,
			IdRef memoryHeight,
			IdRef memoryPitch,
			IdRef coordinate)
		{
			uint16_t wordCount = 10;

			writeWord(spv::Op::OpSubgroup2DBlockPrefetchINTEL, wordCount);
			writeWords(elementSize, blockWidth, blockHeight, blockCount, srcBasePointer, memoryWidth, memoryHeight, memoryPitch, coordinate);
		}

		void OpSubgroup2DBlockStoreINTEL(
			IdRef elementSize,
			IdRef blockWidth,
			IdRef blockHeight,
			IdRef blockCount,
			IdRef srcPointer,
			IdRef dstBasePointer,
			IdRef memoryWidth,
			IdRef memoryHeight,
			IdRef memoryPitch,
			IdRef coordinate)
		{
			uint16_t wordCount = 11;

			writeWord(spv::Op::OpSubgroup2DBlockStoreINTEL, wordCount);
			writeWords(elementSize, blockWidth, blockHeight, blockCount, srcPointer, dstBasePointer, memoryWidth, memoryHeight, memoryPitch, coordinate);
		}

		void OpSubgroupAllEqualKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef predicate)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAllEqualKHR, wordCount);
			writeWords(idResultType, idResult, predicate);
		}

		void OpSubgroupAllKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef predicate)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAllKHR, wordCount);
			writeWords(idResultType, idResult, predicate);
		}

		void OpSubgroupAnyKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef predicate)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAnyKHR, wordCount);
			writeWords(idResultType, idResult, predicate);
		}

		void OpSubgroupAvcBmeInitializeINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef srcCoord,
			IdRef motionVectors,
			IdRef majorShapes,
			IdRef minorShapes,
			IdRef direction,
			IdRef pixelResolution,
			IdRef bidirectionalWeight,
			IdRef sadAdjustment)
		{
			uint16_t wordCount = 11;

			writeWord(spv::Op::OpSubgroupAvcBmeInitializeINTEL, wordCount);
			writeWords(idResultType, idResult, srcCoord, motionVectors, majorShapes, minorShapes, direction, pixelResolution, bidirectionalWeight, sadAdjustment);
		}

		void OpSubgroupAvcFmeInitializeINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef srcCoord,
			IdRef motionVectors,
			IdRef majorShapes,
			IdRef minorShapes,
			IdRef direction,
			IdRef pixelResolution,
			IdRef sadAdjustment)
		{
			uint16_t wordCount = 10;

			writeWord(spv::Op::OpSubgroupAvcFmeInitializeINTEL, wordCount);
			writeWords(idResultType, idResult, srcCoord, motionVectors, majorShapes, minorShapes, direction, pixelResolution, sadAdjustment);
		}

		void OpSubgroupAvcImeAdjustRefOffsetINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef refOffset,
			IdRef srcCoord,
			IdRef refWindowSize,
			IdRef imageSize)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpSubgroupAvcImeAdjustRefOffsetINTEL, wordCount);
			writeWords(idResultType, idResult, refOffset, srcCoord, refWindowSize, imageSize);
		}

		void OpSubgroupAvcImeConvertToMcePayloadINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcImeConvertToMcePayloadINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcImeConvertToMceResultINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcImeConvertToMceResultINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcImeEvaluateWithDualReferenceINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef srcImage,
			IdRef fwdRefImage,
			IdRef bwdRefImage,
			IdRef payload)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpSubgroupAvcImeEvaluateWithDualReferenceINTEL, wordCount);
			writeWords(idResultType, idResult, srcImage, fwdRefImage, bwdRefImage, payload);
		}

		void OpSubgroupAvcImeEvaluateWithDualReferenceStreaminINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef srcImage,
			IdRef fwdRefImage,
			IdRef bwdRefImage,
			IdRef payload,
			IdRef streaminComponents)
		{
			uint16_t wordCount = 8;

			writeWord(spv::Op::OpSubgroupAvcImeEvaluateWithDualReferenceStreaminINTEL, wordCount);
			writeWords(idResultType, idResult, srcImage, fwdRefImage, bwdRefImage, payload, streaminComponents);
		}

		void OpSubgroupAvcImeEvaluateWithDualReferenceStreaminoutINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef srcImage,
			IdRef fwdRefImage,
			IdRef bwdRefImage,
			IdRef payload,
			IdRef streaminComponents)
		{
			uint16_t wordCount = 8;

			writeWord(spv::Op::OpSubgroupAvcImeEvaluateWithDualReferenceStreaminoutINTEL, wordCount);
			writeWords(idResultType, idResult, srcImage, fwdRefImage, bwdRefImage, payload, streaminComponents);
		}

		void OpSubgroupAvcImeEvaluateWithDualReferenceStreamoutINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef srcImage,
			IdRef fwdRefImage,
			IdRef bwdRefImage,
			IdRef payload)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpSubgroupAvcImeEvaluateWithDualReferenceStreamoutINTEL, wordCount);
			writeWords(idResultType, idResult, srcImage, fwdRefImage, bwdRefImage, payload);
		}

		void OpSubgroupAvcImeEvaluateWithSingleReferenceINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef srcImage,
			IdRef refImage,
			IdRef payload)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpSubgroupAvcImeEvaluateWithSingleReferenceINTEL, wordCount);
			writeWords(idResultType, idResult, srcImage, refImage, payload);
		}

		void OpSubgroupAvcImeEvaluateWithSingleReferenceStreaminINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef srcImage,
			IdRef refImage,
			IdRef payload,
			IdRef streaminComponents)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpSubgroupAvcImeEvaluateWithSingleReferenceStreaminINTEL, wordCount);
			writeWords(idResultType, idResult, srcImage, refImage, payload, streaminComponents);
		}

		void OpSubgroupAvcImeEvaluateWithSingleReferenceStreaminoutINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef srcImage,
			IdRef refImage,
			IdRef payload,
			IdRef streaminComponents)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpSubgroupAvcImeEvaluateWithSingleReferenceStreaminoutINTEL, wordCount);
			writeWords(idResultType, idResult, srcImage, refImage, payload, streaminComponents);
		}

		void OpSubgroupAvcImeEvaluateWithSingleReferenceStreamoutINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef srcImage,
			IdRef refImage,
			IdRef payload)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpSubgroupAvcImeEvaluateWithSingleReferenceStreamoutINTEL, wordCount);
			writeWords(idResultType, idResult, srcImage, refImage, payload);
		}

		void OpSubgroupAvcImeGetBorderReachedINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef imageSelect,
			IdRef payload)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcImeGetBorderReachedINTEL, wordCount);
			writeWords(idResultType, idResult, imageSelect, payload);
		}

		void OpSubgroupAvcImeGetDualReferenceStreaminINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcImeGetDualReferenceStreaminINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcImeGetSingleReferenceStreaminINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcImeGetSingleReferenceStreaminINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeDistortionsINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload,
			IdRef majorShape,
			IdRef direction)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeDistortionsINTEL, wordCount);
			writeWords(idResultType, idResult, payload, majorShape, direction);
		}

		void OpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeMotionVectorsINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload,
			IdRef majorShape,
			IdRef direction)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeMotionVectorsINTEL, wordCount);
			writeWords(idResultType, idResult, payload, majorShape, direction);
		}

		void OpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeReferenceIdsINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload,
			IdRef majorShape,
			IdRef direction)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeReferenceIdsINTEL, wordCount);
			writeWords(idResultType, idResult, payload, majorShape, direction);
		}

		void OpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeDistortionsINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload,
			IdRef majorShape)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeDistortionsINTEL, wordCount);
			writeWords(idResultType, idResult, payload, majorShape);
		}

		void OpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeMotionVectorsINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload,
			IdRef majorShape)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeMotionVectorsINTEL, wordCount);
			writeWords(idResultType, idResult, payload, majorShape);
		}

		void OpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeReferenceIdsINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload,
			IdRef majorShape)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeReferenceIdsINTEL, wordCount);
			writeWords(idResultType, idResult, payload, majorShape);
		}

		void OpSubgroupAvcImeGetTruncatedSearchIndicationINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcImeGetTruncatedSearchIndicationINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcImeGetUnidirectionalEarlySearchTerminationINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcImeGetUnidirectionalEarlySearchTerminationINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcImeGetWeightingPatternMinimumDistortionINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcImeGetWeightingPatternMinimumDistortionINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcImeGetWeightingPatternMinimumMotionVectorINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcImeGetWeightingPatternMinimumMotionVectorINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcImeInitializeINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef srcCoord,
			IdRef partitionMask,
			IdRef sADAdjustment)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpSubgroupAvcImeInitializeINTEL, wordCount);
			writeWords(idResultType, idResult, srcCoord, partitionMask, sADAdjustment);
		}

		void OpSubgroupAvcImeRefWindowSizeINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef searchWindowConfig,
			IdRef dualRef)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcImeRefWindowSizeINTEL, wordCount);
			writeWords(idResultType, idResult, searchWindowConfig, dualRef);
		}

		void OpSubgroupAvcImeSetDualReferenceINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef fwdRefOffset,
			IdRef bwdRefOffset,
			IdRef idSearchWindowConfig,
			IdRef payload)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpSubgroupAvcImeSetDualReferenceINTEL, wordCount);
			writeWords(idResultType, idResult, fwdRefOffset, bwdRefOffset, idSearchWindowConfig, payload);
		}

		void OpSubgroupAvcImeSetEarlySearchTerminationThresholdINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef threshold,
			IdRef payload)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcImeSetEarlySearchTerminationThresholdINTEL, wordCount);
			writeWords(idResultType, idResult, threshold, payload);
		}

		void OpSubgroupAvcImeSetMaxMotionVectorCountINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef maxMotionVectorCount,
			IdRef payload)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcImeSetMaxMotionVectorCountINTEL, wordCount);
			writeWords(idResultType, idResult, maxMotionVectorCount, payload);
		}

		void OpSubgroupAvcImeSetSingleReferenceINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef refOffset,
			IdRef searchWindowConfig,
			IdRef payload)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpSubgroupAvcImeSetSingleReferenceINTEL, wordCount);
			writeWords(idResultType, idResult, refOffset, searchWindowConfig, payload);
		}

		void OpSubgroupAvcImeSetUnidirectionalMixDisableINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcImeSetUnidirectionalMixDisableINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcImeSetWeightedSadINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef packedSadWeights,
			IdRef payload)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcImeSetWeightedSadINTEL, wordCount);
			writeWords(idResultType, idResult, packedSadWeights, payload);
		}

		void OpSubgroupAvcImeStripDualReferenceStreamoutINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcImeStripDualReferenceStreamoutINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcImeStripSingleReferenceStreamoutINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcImeStripSingleReferenceStreamoutINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcMceConvertToImePayloadINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcMceConvertToImePayloadINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcMceConvertToImeResultINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcMceConvertToImeResultINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcMceConvertToRefPayloadINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcMceConvertToRefPayloadINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcMceConvertToRefResultINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcMceConvertToRefResultINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcMceConvertToSicPayloadINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcMceConvertToSicPayloadINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcMceConvertToSicResultINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcMceConvertToSicResultINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcMceGetBestInterDistortionsINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcMceGetBestInterDistortionsINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcMceGetDefaultHighPenaltyCostTableINTEL(
			IdResultType idResultType,
			IdResult idResult)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpSubgroupAvcMceGetDefaultHighPenaltyCostTableINTEL, wordCount);
			writeWords(idResultType, idResult);
		}

		void OpSubgroupAvcMceGetDefaultInterBaseMultiReferencePenaltyINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sliceType,
			IdRef qp)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcMceGetDefaultInterBaseMultiReferencePenaltyINTEL, wordCount);
			writeWords(idResultType, idResult, sliceType, qp);
		}

		void OpSubgroupAvcMceGetDefaultInterDirectionPenaltyINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sliceType,
			IdRef qp)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcMceGetDefaultInterDirectionPenaltyINTEL, wordCount);
			writeWords(idResultType, idResult, sliceType, qp);
		}

		void OpSubgroupAvcMceGetDefaultInterMotionVectorCostTableINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sliceType,
			IdRef qp)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcMceGetDefaultInterMotionVectorCostTableINTEL, wordCount);
			writeWords(idResultType, idResult, sliceType, qp);
		}

		void OpSubgroupAvcMceGetDefaultInterShapePenaltyINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sliceType,
			IdRef qp)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcMceGetDefaultInterShapePenaltyINTEL, wordCount);
			writeWords(idResultType, idResult, sliceType, qp);
		}

		void OpSubgroupAvcMceGetDefaultIntraChromaModeBasePenaltyINTEL(
			IdResultType idResultType,
			IdResult idResult)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpSubgroupAvcMceGetDefaultIntraChromaModeBasePenaltyINTEL, wordCount);
			writeWords(idResultType, idResult);
		}

		void OpSubgroupAvcMceGetDefaultIntraLumaModePenaltyINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sliceType,
			IdRef qp)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcMceGetDefaultIntraLumaModePenaltyINTEL, wordCount);
			writeWords(idResultType, idResult, sliceType, qp);
		}

		void OpSubgroupAvcMceGetDefaultIntraLumaShapePenaltyINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sliceType,
			IdRef qp)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcMceGetDefaultIntraLumaShapePenaltyINTEL, wordCount);
			writeWords(idResultType, idResult, sliceType, qp);
		}

		void OpSubgroupAvcMceGetDefaultLowPenaltyCostTableINTEL(
			IdResultType idResultType,
			IdResult idResult)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpSubgroupAvcMceGetDefaultLowPenaltyCostTableINTEL, wordCount);
			writeWords(idResultType, idResult);
		}

		void OpSubgroupAvcMceGetDefaultMediumPenaltyCostTableINTEL(
			IdResultType idResultType,
			IdResult idResult)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpSubgroupAvcMceGetDefaultMediumPenaltyCostTableINTEL, wordCount);
			writeWords(idResultType, idResult);
		}

		void OpSubgroupAvcMceGetDefaultNonDcLumaIntraPenaltyINTEL(
			IdResultType idResultType,
			IdResult idResult)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpSubgroupAvcMceGetDefaultNonDcLumaIntraPenaltyINTEL, wordCount);
			writeWords(idResultType, idResult);
		}

		void OpSubgroupAvcMceGetInterDirectionsINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcMceGetInterDirectionsINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcMceGetInterDistortionsINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcMceGetInterDistortionsINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcMceGetInterMajorShapeINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcMceGetInterMajorShapeINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcMceGetInterMinorShapeINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcMceGetInterMinorShapeINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcMceGetInterMotionVectorCountINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcMceGetInterMotionVectorCountINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcMceGetInterReferenceIdsINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcMceGetInterReferenceIdsINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcMceGetInterReferenceInterlacedFieldPolaritiesINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef packedReferenceIds,
			IdRef packedReferenceParameterFieldPolarities,
			IdRef payload)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpSubgroupAvcMceGetInterReferenceInterlacedFieldPolaritiesINTEL, wordCount);
			writeWords(idResultType, idResult, packedReferenceIds, packedReferenceParameterFieldPolarities, payload);
		}

		void OpSubgroupAvcMceGetMotionVectorsINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcMceGetMotionVectorsINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcMceSetAcOnlyHaarINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcMceSetAcOnlyHaarINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcMceSetDualReferenceInterlacedFieldPolaritiesINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef forwardReferenceFieldPolarity,
			IdRef backwardReferenceFieldPolarity,
			IdRef payload)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpSubgroupAvcMceSetDualReferenceInterlacedFieldPolaritiesINTEL, wordCount);
			writeWords(idResultType, idResult, forwardReferenceFieldPolarity, backwardReferenceFieldPolarity, payload);
		}

		void OpSubgroupAvcMceSetInterBaseMultiReferencePenaltyINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef referenceBasePenalty,
			IdRef payload)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcMceSetInterBaseMultiReferencePenaltyINTEL, wordCount);
			writeWords(idResultType, idResult, referenceBasePenalty, payload);
		}

		void OpSubgroupAvcMceSetInterDirectionPenaltyINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef directionCost,
			IdRef payload)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcMceSetInterDirectionPenaltyINTEL, wordCount);
			writeWords(idResultType, idResult, directionCost, payload);
		}

		void OpSubgroupAvcMceSetInterShapePenaltyINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef packedShapePenalty,
			IdRef payload)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcMceSetInterShapePenaltyINTEL, wordCount);
			writeWords(idResultType, idResult, packedShapePenalty, payload);
		}

		void OpSubgroupAvcMceSetMotionVectorCostFunctionINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef packedCostCenterDelta,
			IdRef packedCostTable,
			IdRef costPrecision,
			IdRef payload)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpSubgroupAvcMceSetMotionVectorCostFunctionINTEL, wordCount);
			writeWords(idResultType, idResult, packedCostCenterDelta, packedCostTable, costPrecision, payload);
		}

		void OpSubgroupAvcMceSetSingleReferenceInterlacedFieldPolarityINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef referenceFieldPolarity,
			IdRef payload)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcMceSetSingleReferenceInterlacedFieldPolarityINTEL, wordCount);
			writeWords(idResultType, idResult, referenceFieldPolarity, payload);
		}

		void OpSubgroupAvcMceSetSourceInterlacedFieldPolarityINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sourceFieldPolarity,
			IdRef payload)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcMceSetSourceInterlacedFieldPolarityINTEL, wordCount);
			writeWords(idResultType, idResult, sourceFieldPolarity, payload);
		}

		void OpSubgroupAvcRefConvertToMcePayloadINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcRefConvertToMcePayloadINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcRefConvertToMceResultINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcRefConvertToMceResultINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcRefEvaluateWithDualReferenceINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef srcImage,
			IdRef fwdRefImage,
			IdRef bwdRefImage,
			IdRef payload)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpSubgroupAvcRefEvaluateWithDualReferenceINTEL, wordCount);
			writeWords(idResultType, idResult, srcImage, fwdRefImage, bwdRefImage, payload);
		}

		void OpSubgroupAvcRefEvaluateWithMultiReferenceINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef srcImage,
			IdRef packedReferenceIds,
			IdRef payload)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpSubgroupAvcRefEvaluateWithMultiReferenceINTEL, wordCount);
			writeWords(idResultType, idResult, srcImage, packedReferenceIds, payload);
		}

		void OpSubgroupAvcRefEvaluateWithMultiReferenceInterlacedINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef srcImage,
			IdRef packedReferenceIds,
			IdRef packedReferenceFieldPolarities,
			IdRef payload)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpSubgroupAvcRefEvaluateWithMultiReferenceInterlacedINTEL, wordCount);
			writeWords(idResultType, idResult, srcImage, packedReferenceIds, packedReferenceFieldPolarities, payload);
		}

		void OpSubgroupAvcRefEvaluateWithSingleReferenceINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef srcImage,
			IdRef refImage,
			IdRef payload)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpSubgroupAvcRefEvaluateWithSingleReferenceINTEL, wordCount);
			writeWords(idResultType, idResult, srcImage, refImage, payload);
		}

		void OpSubgroupAvcRefSetBidirectionalMixDisableINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcRefSetBidirectionalMixDisableINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcRefSetBilinearFilterEnableINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcRefSetBilinearFilterEnableINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcSicConfigureIpeLumaChromaINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef lumaIntraPartitionMask,
			IdRef intraNeighbourAvailabilty,
			IdRef leftEdgeLumaPixels,
			IdRef upperLeftCornerLumaPixel,
			IdRef upperEdgeLumaPixels,
			IdRef upperRightEdgeLumaPixels,
			IdRef leftEdgeChromaPixels,
			IdRef upperLeftCornerChromaPixel,
			IdRef upperEdgeChromaPixels,
			IdRef sadAdjustment,
			IdRef payload)
		{
			uint16_t wordCount = 14;

			writeWord(spv::Op::OpSubgroupAvcSicConfigureIpeLumaChromaINTEL, wordCount);
			writeWords(idResultType, idResult, lumaIntraPartitionMask, intraNeighbourAvailabilty, leftEdgeLumaPixels, upperLeftCornerLumaPixel, upperEdgeLumaPixels, upperRightEdgeLumaPixels, leftEdgeChromaPixels, upperLeftCornerChromaPixel, upperEdgeChromaPixels, sadAdjustment, payload);
		}

		void OpSubgroupAvcSicConfigureIpeLumaINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef lumaIntraPartitionMask,
			IdRef intraNeighbourAvailabilty,
			IdRef leftEdgeLumaPixels,
			IdRef upperLeftCornerLumaPixel,
			IdRef upperEdgeLumaPixels,
			IdRef upperRightEdgeLumaPixels,
			IdRef sadAdjustment,
			IdRef payload)
		{
			uint16_t wordCount = 11;

			writeWord(spv::Op::OpSubgroupAvcSicConfigureIpeLumaINTEL, wordCount);
			writeWords(idResultType, idResult, lumaIntraPartitionMask, intraNeighbourAvailabilty, leftEdgeLumaPixels, upperLeftCornerLumaPixel, upperEdgeLumaPixels, upperRightEdgeLumaPixels, sadAdjustment, payload);
		}

		void OpSubgroupAvcSicConfigureSkcINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef skipBlockPartitionType,
			IdRef skipMotionVectorMask,
			IdRef motionVectors,
			IdRef bidirectionalWeight,
			IdRef sadAdjustment,
			IdRef payload)
		{
			uint16_t wordCount = 9;

			writeWord(spv::Op::OpSubgroupAvcSicConfigureSkcINTEL, wordCount);
			writeWords(idResultType, idResult, skipBlockPartitionType, skipMotionVectorMask, motionVectors, bidirectionalWeight, sadAdjustment, payload);
		}

		void OpSubgroupAvcSicConvertToMcePayloadINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcSicConvertToMcePayloadINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcSicConvertToMceResultINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcSicConvertToMceResultINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcSicEvaluateIpeINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef srcImage,
			IdRef payload)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcSicEvaluateIpeINTEL, wordCount);
			writeWords(idResultType, idResult, srcImage, payload);
		}

		void OpSubgroupAvcSicEvaluateWithDualReferenceINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef srcImage,
			IdRef fwdRefImage,
			IdRef bwdRefImage,
			IdRef payload)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpSubgroupAvcSicEvaluateWithDualReferenceINTEL, wordCount);
			writeWords(idResultType, idResult, srcImage, fwdRefImage, bwdRefImage, payload);
		}

		void OpSubgroupAvcSicEvaluateWithMultiReferenceINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef srcImage,
			IdRef packedReferenceIds,
			IdRef payload)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpSubgroupAvcSicEvaluateWithMultiReferenceINTEL, wordCount);
			writeWords(idResultType, idResult, srcImage, packedReferenceIds, payload);
		}

		void OpSubgroupAvcSicEvaluateWithMultiReferenceInterlacedINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef srcImage,
			IdRef packedReferenceIds,
			IdRef packedReferenceFieldPolarities,
			IdRef payload)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpSubgroupAvcSicEvaluateWithMultiReferenceInterlacedINTEL, wordCount);
			writeWords(idResultType, idResult, srcImage, packedReferenceIds, packedReferenceFieldPolarities, payload);
		}

		void OpSubgroupAvcSicEvaluateWithSingleReferenceINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef srcImage,
			IdRef refImage,
			IdRef payload)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpSubgroupAvcSicEvaluateWithSingleReferenceINTEL, wordCount);
			writeWords(idResultType, idResult, srcImage, refImage, payload);
		}

		void OpSubgroupAvcSicGetBestIpeChromaDistortionINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcSicGetBestIpeChromaDistortionINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcSicGetBestIpeLumaDistortionINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcSicGetBestIpeLumaDistortionINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcSicGetInterRawSadsINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcSicGetInterRawSadsINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcSicGetIpeChromaModeINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcSicGetIpeChromaModeINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcSicGetIpeLumaShapeINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcSicGetIpeLumaShapeINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcSicGetMotionVectorMaskINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef skipBlockPartitionType,
			IdRef direction)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcSicGetMotionVectorMaskINTEL, wordCount);
			writeWords(idResultType, idResult, skipBlockPartitionType, direction);
		}

		void OpSubgroupAvcSicGetPackedIpeLumaModesINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcSicGetPackedIpeLumaModesINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcSicGetPackedSkcLumaCountThresholdINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcSicGetPackedSkcLumaCountThresholdINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcSicGetPackedSkcLumaSumThresholdINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcSicGetPackedSkcLumaSumThresholdINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcSicInitializeINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef srcCoord)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcSicInitializeINTEL, wordCount);
			writeWords(idResultType, idResult, srcCoord);
		}

		void OpSubgroupAvcSicSetBilinearFilterEnableINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef payload)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupAvcSicSetBilinearFilterEnableINTEL, wordCount);
			writeWords(idResultType, idResult, payload);
		}

		void OpSubgroupAvcSicSetBlockBasedRawSkipSadINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef blockBasedSkipType,
			IdRef payload)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcSicSetBlockBasedRawSkipSadINTEL, wordCount);
			writeWords(idResultType, idResult, blockBasedSkipType, payload);
		}

		void OpSubgroupAvcSicSetIntraChromaModeCostFunctionINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef chromaModeBasePenalty,
			IdRef payload)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcSicSetIntraChromaModeCostFunctionINTEL, wordCount);
			writeWords(idResultType, idResult, chromaModeBasePenalty, payload);
		}

		void OpSubgroupAvcSicSetIntraLumaModeCostFunctionINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef lumaModePenalty,
			IdRef lumaPackedNeighborModes,
			IdRef lumaPackedNonDcPenalty,
			IdRef payload)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpSubgroupAvcSicSetIntraLumaModeCostFunctionINTEL, wordCount);
			writeWords(idResultType, idResult, lumaModePenalty, lumaPackedNeighborModes, lumaPackedNonDcPenalty, payload);
		}

		void OpSubgroupAvcSicSetIntraLumaShapePenaltyINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef packedShapePenalty,
			IdRef payload)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcSicSetIntraLumaShapePenaltyINTEL, wordCount);
			writeWords(idResultType, idResult, packedShapePenalty, payload);
		}

		void OpSubgroupAvcSicSetSkcForwardTransformEnableINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef packedSadCoefficients,
			IdRef payload)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupAvcSicSetSkcForwardTransformEnableINTEL, wordCount);
			writeWords(idResultType, idResult, packedSadCoefficients, payload);
		}

		void OpSubgroupBallotKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef predicate)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupBallotKHR, wordCount);
			writeWords(idResultType, idResult, predicate);
		}

		void OpSubgroupBlockPrefetchINTEL(
			IdRef ptr,
			IdRef numBytes,
			std::optional<spv::MemoryAccessMask> memoryAccess = {})
		{
			uint16_t wordCount = 3;
			countOperandsWord(wordCount, memoryAccess);

			writeWord(spv::Op::OpSubgroupBlockPrefetchINTEL, wordCount);
			writeWords(ptr, numBytes, memoryAccess);
		}

		void OpSubgroupBlockReadINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef ptr)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupBlockReadINTEL, wordCount);
			writeWords(idResultType, idResult, ptr);
		}

		void OpSubgroupBlockWriteINTEL(
			IdRef ptr,
			IdRef data)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpSubgroupBlockWriteINTEL, wordCount);
			writeWords(ptr, data);
		}

		void OpSubgroupFirstInvocationKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef value)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupFirstInvocationKHR, wordCount);
			writeWords(idResultType, idResult, value);
		}

		void OpSubgroupImageBlockReadINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef image,
			IdRef coordinate)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupImageBlockReadINTEL, wordCount);
			writeWords(idResultType, idResult, image, coordinate);
		}

		void OpSubgroupImageBlockWriteINTEL(
			IdRef image,
			IdRef coordinate,
			IdRef data)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpSubgroupImageBlockWriteINTEL, wordCount);
			writeWords(image, coordinate, data);
		}

		void OpSubgroupImageMediaBlockReadINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef image,
			IdRef coordinate,
			IdRef width,
			IdRef height)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpSubgroupImageMediaBlockReadINTEL, wordCount);
			writeWords(idResultType, idResult, image, coordinate, width, height);
		}

		void OpSubgroupImageMediaBlockWriteINTEL(
			IdRef image,
			IdRef coordinate,
			IdRef width,
			IdRef height,
			IdRef data)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpSubgroupImageMediaBlockWriteINTEL, wordCount);
			writeWords(image, coordinate, width, height, data);
		}

		void OpSubgroupMatrixMultiplyAccumulateINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef kDim,
			IdRef matrixA,
			IdRef matrixB,
			IdRef matrixC,
			std::optional<spv::MatrixMultiplyAccumulateOperandsMask> matrixMultiplyAccumulateOperands = {})
		{
			uint16_t wordCount = 7;
			countOperandsWord(wordCount, matrixMultiplyAccumulateOperands);

			writeWord(spv::Op::OpSubgroupMatrixMultiplyAccumulateINTEL, wordCount);
			writeWords(idResultType, idResult, kDim, matrixA, matrixB, matrixC, matrixMultiplyAccumulateOperands);
		}

		void OpSubgroupReadInvocationKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef value,
			IdRef index)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupReadInvocationKHR, wordCount);
			writeWords(idResultType, idResult, value, index);
		}

		void OpSubgroupShuffleDownINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef current,
			IdRef next,
			IdRef delta)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpSubgroupShuffleDownINTEL, wordCount);
			writeWords(idResultType, idResult, current, next, delta);
		}

		void OpSubgroupShuffleINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef data,
			IdRef invocationId)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupShuffleINTEL, wordCount);
			writeWords(idResultType, idResult, data, invocationId);
		}

		void OpSubgroupShuffleUpINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef previous,
			IdRef current,
			IdRef delta)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpSubgroupShuffleUpINTEL, wordCount);
			writeWords(idResultType, idResult, previous, current, delta);
		}

		void OpSubgroupShuffleXorINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef data,
			IdRef value)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpSubgroupShuffleXorINTEL, wordCount);
			writeWords(idResultType, idResult, data, value);
		}

		void OpSwitch(
			IdRef selector,
			IdRef _default,
			const std::vector<std::tuple<uint32_t, IdRef>>& target = {})
		{
			uint16_t wordCount = 3;
			countOperandsWord(wordCount, target);

			writeWord(spv::Op::OpSwitch, wordCount);
			writeWords(selector, _default, target);
		}

		void OpTaskSequenceAsyncINTEL(
			IdRef sequence,
			const std::vector<IdRef>& arguments = {})
		{
			uint16_t wordCount = 2;
			countOperandsWord(wordCount, arguments);

			writeWord(spv::Op::OpTaskSequenceAsyncINTEL, wordCount);
			writeWords(sequence, arguments);
		}

		void OpTaskSequenceCreateINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef function,
			uint32_t pipelined,
			uint32_t useStallEnableClusters,
			uint32_t getCapacity,
			uint32_t asyncCapacity)
		{
			uint16_t wordCount = 8;

			writeWord(spv::Op::OpTaskSequenceCreateINTEL, wordCount);
			writeWords(idResultType, idResult, function, pipelined, useStallEnableClusters, getCapacity, asyncCapacity);
		}

		void OpTaskSequenceGetINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef sequence)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpTaskSequenceGetINTEL, wordCount);
			writeWords(idResultType, idResult, sequence);
		}

		void OpTaskSequenceReleaseINTEL(IdRef sequence)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTaskSequenceReleaseINTEL, wordCount);
			writeWords(sequence);
		}

		void OpTensorLayoutSetBlockSizeNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef tensorLayout,
			const std::vector<IdRef>& blockSize = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, blockSize);

			writeWord(spv::Op::OpTensorLayoutSetBlockSizeNV, wordCount);
			writeWords(idResultType, idResult, tensorLayout, blockSize);
		}

		void OpTensorLayoutSetClampValueNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef tensorLayout,
			IdRef value)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpTensorLayoutSetClampValueNV, wordCount);
			writeWords(idResultType, idResult, tensorLayout, value);
		}

		void OpTensorLayoutSetDimensionNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef tensorLayout,
			const std::vector<IdRef>& dim = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, dim);

			writeWord(spv::Op::OpTensorLayoutSetDimensionNV, wordCount);
			writeWords(idResultType, idResult, tensorLayout, dim);
		}

		void OpTensorLayoutSetStrideNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef tensorLayout,
			const std::vector<IdRef>& stride = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, stride);

			writeWord(spv::Op::OpTensorLayoutSetStrideNV, wordCount);
			writeWords(idResultType, idResult, tensorLayout, stride);
		}

		void OpTensorLayoutSliceNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef tensorLayout,
			const std::vector<IdRef>& operands = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, operands);

			writeWord(spv::Op::OpTensorLayoutSliceNV, wordCount);
			writeWords(idResultType, idResult, tensorLayout, operands);
		}

		void OpTensorViewSetClipNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef tensorView,
			IdRef clipRowOffset,
			IdRef clipRowSpan,
			IdRef clipColOffset,
			IdRef clipColSpan)
		{
			uint16_t wordCount = 8;

			writeWord(spv::Op::OpTensorViewSetClipNV, wordCount);
			writeWords(idResultType, idResult, tensorView, clipRowOffset, clipRowSpan, clipColOffset, clipColSpan);
		}

		void OpTensorViewSetDimensionNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef tensorView,
			const std::vector<IdRef>& dim = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, dim);

			writeWord(spv::Op::OpTensorViewSetDimensionNV, wordCount);
			writeWords(idResultType, idResult, tensorView, dim);
		}

		void OpTensorViewSetStrideNV(
			IdResultType idResultType,
			IdResult idResult,
			IdRef tensorView,
			const std::vector<IdRef>& stride = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, stride);

			writeWord(spv::Op::OpTensorViewSetStrideNV, wordCount);
			writeWords(idResultType, idResult, tensorView, stride);
		}

		void OpTerminateInvocation()
		{
			uint16_t wordCount = 1;

			writeWord(spv::Op::OpTerminateInvocation, wordCount);
			writeWords();
		}

		void OpTerminateRayKHR()
		{
			uint16_t wordCount = 1;

			writeWord(spv::Op::OpTerminateRayKHR, wordCount);
			writeWords();
		}

		void OpTerminateRayNV()
		{
			uint16_t wordCount = 1;

			writeWord(spv::Op::OpTerminateRayNV, wordCount);
			writeWords();
		}

		void OpTraceMotionNV(
			IdRef accel,
			IdRef rayFlags,
			IdRef cullMask,
			IdRef sBTOffset,
			IdRef sBTStride,
			IdRef missIndex,
			IdRef rayOrigin,
			IdRef rayTmin,
			IdRef rayDirection,
			IdRef rayTmax,
			IdRef time,
			IdRef payloadId)
		{
			uint16_t wordCount = 13;

			writeWord(spv::Op::OpTraceMotionNV, wordCount);
			writeWords(accel, rayFlags, cullMask, sBTOffset, sBTStride, missIndex, rayOrigin, rayTmin, rayDirection, rayTmax, time, payloadId);
		}

		void OpTraceNV(
			IdRef accel,
			IdRef rayFlags,
			IdRef cullMask,
			IdRef sBTOffset,
			IdRef sBTStride,
			IdRef missIndex,
			IdRef rayOrigin,
			IdRef rayTmin,
			IdRef rayDirection,
			IdRef rayTmax,
			IdRef payloadId)
		{
			uint16_t wordCount = 12;

			writeWord(spv::Op::OpTraceNV, wordCount);
			writeWords(accel, rayFlags, cullMask, sBTOffset, sBTStride, missIndex, rayOrigin, rayTmin, rayDirection, rayTmax, payloadId);
		}

		void OpTraceRayKHR(
			IdRef accel,
			IdRef rayFlags,
			IdRef cullMask,
			IdRef sBTOffset,
			IdRef sBTStride,
			IdRef missIndex,
			IdRef rayOrigin,
			IdRef rayTmin,
			IdRef rayDirection,
			IdRef rayTmax,
			IdRef payload)
		{
			uint16_t wordCount = 12;

			writeWord(spv::Op::OpTraceRayKHR, wordCount);
			writeWords(accel, rayFlags, cullMask, sBTOffset, sBTStride, missIndex, rayOrigin, rayTmin, rayDirection, rayTmax, payload);
		}

		void OpTraceRayMotionNV(
			IdRef accel,
			IdRef rayFlags,
			IdRef cullMask,
			IdRef sBTOffset,
			IdRef sBTStride,
			IdRef missIndex,
			IdRef rayOrigin,
			IdRef rayTmin,
			IdRef rayDirection,
			IdRef rayTmax,
			IdRef time,
			IdRef payload)
		{
			uint16_t wordCount = 13;

			writeWord(spv::Op::OpTraceRayMotionNV, wordCount);
			writeWords(accel, rayFlags, cullMask, sBTOffset, sBTStride, missIndex, rayOrigin, rayTmin, rayDirection, rayTmax, time, payload);
		}

		void OpTranspose(
			IdResultType idResultType,
			IdResult idResult,
			IdRef matrix)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpTranspose, wordCount);
			writeWords(idResultType, idResult, matrix);
		}

		void OpTypeAccelerationStructureKHR(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeAccelerationStructureKHR, wordCount);
			writeWords(idResult);
		}

		void OpTypeArray(
			IdResult idResult,
			IdRef elementType,
			IdRef length)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpTypeArray, wordCount);
			writeWords(idResult, elementType, length);
		}

		void OpTypeAvcImeDualReferenceStreaminINTEL(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeAvcImeDualReferenceStreaminINTEL, wordCount);
			writeWords(idResult);
		}

		void OpTypeAvcImePayloadINTEL(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeAvcImePayloadINTEL, wordCount);
			writeWords(idResult);
		}

		void OpTypeAvcImeResultDualReferenceStreamoutINTEL(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeAvcImeResultDualReferenceStreamoutINTEL, wordCount);
			writeWords(idResult);
		}

		void OpTypeAvcImeResultINTEL(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeAvcImeResultINTEL, wordCount);
			writeWords(idResult);
		}

		void OpTypeAvcImeResultSingleReferenceStreamoutINTEL(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeAvcImeResultSingleReferenceStreamoutINTEL, wordCount);
			writeWords(idResult);
		}

		void OpTypeAvcImeSingleReferenceStreaminINTEL(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeAvcImeSingleReferenceStreaminINTEL, wordCount);
			writeWords(idResult);
		}

		void OpTypeAvcMcePayloadINTEL(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeAvcMcePayloadINTEL, wordCount);
			writeWords(idResult);
		}

		void OpTypeAvcMceResultINTEL(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeAvcMceResultINTEL, wordCount);
			writeWords(idResult);
		}

		void OpTypeAvcRefPayloadINTEL(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeAvcRefPayloadINTEL, wordCount);
			writeWords(idResult);
		}

		void OpTypeAvcRefResultINTEL(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeAvcRefResultINTEL, wordCount);
			writeWords(idResult);
		}

		void OpTypeAvcSicPayloadINTEL(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeAvcSicPayloadINTEL, wordCount);
			writeWords(idResult);
		}

		void OpTypeAvcSicResultINTEL(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeAvcSicResultINTEL, wordCount);
			writeWords(idResult);
		}

		void OpTypeBool(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeBool, wordCount);
			writeWords(idResult);
		}

		void OpTypeBufferSurfaceINTEL(
			IdResult idResult,
			spv::AccessQualifier accessQualifier)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpTypeBufferSurfaceINTEL, wordCount);
			writeWords(idResult, accessQualifier);
		}

		void OpTypeCooperativeMatrixKHR(
			IdResult idResult,
			IdRef componentType,
			IdScope scope,
			IdRef rows,
			IdRef columns,
			IdRef use)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpTypeCooperativeMatrixKHR, wordCount);
			writeWords(idResult, componentType, scope, rows, columns, use);
		}

		void OpTypeCooperativeMatrixNV(
			IdResult idResult,
			IdRef componentType,
			IdScope execution,
			IdRef rows,
			IdRef columns)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpTypeCooperativeMatrixNV, wordCount);
			writeWords(idResult, componentType, execution, rows, columns);
		}

		void OpTypeCooperativeVectorNV(
			IdResult idResult,
			IdRef componentType,
			IdRef componentCount)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpTypeCooperativeVectorNV, wordCount);
			writeWords(idResult, componentType, componentCount);
		}

		void OpTypeDeviceEvent(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeDeviceEvent, wordCount);
			writeWords(idResult);
		}

		void OpTypeEvent(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeEvent, wordCount);
			writeWords(idResult);
		}

		void OpTypeFloat(
			IdResult idResult,
			uint32_t width,
			std::optional<spv::FPEncoding> floatingPointEncoding = {})
		{
			uint16_t wordCount = 3;
			countOperandsWord(wordCount, floatingPointEncoding);

			writeWord(spv::Op::OpTypeFloat, wordCount);
			writeWords(idResult, width, floatingPointEncoding);
		}

		void OpTypeForwardPointer(
			IdRef pointerType,
			spv::StorageClass storageClass)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpTypeForwardPointer, wordCount);
			writeWords(pointerType, storageClass);
		}

		void OpTypeFunction(
			IdResult idResult,
			IdRef returnType,
			const std::vector<IdRef>& parameterTypes = {})
		{
			uint16_t wordCount = 3;
			countOperandsWord(wordCount, parameterTypes);

			writeWord(spv::Op::OpTypeFunction, wordCount);
			writeWords(idResult, returnType, parameterTypes);
		}

		void OpTypeHitObjectNV(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeHitObjectNV, wordCount);
			writeWords(idResult);
		}

		void OpTypeImage(
			IdResult idResult,
			IdRef sampledType,
			spv::Dim dim,
			uint32_t depth,
			uint32_t arrayed,
			uint32_t MS,
			uint32_t sampled,
			spv::ImageFormat imageFormat,
			std::optional<spv::AccessQualifier> accessQualifier = {})
		{
			uint16_t wordCount = 9;
			countOperandsWord(wordCount, accessQualifier);

			writeWord(spv::Op::OpTypeImage, wordCount);
			writeWords(idResult, sampledType, dim, depth, arrayed, MS, sampled, imageFormat, accessQualifier);
		}

		void OpTypeInt(
			IdResult idResult,
			uint32_t width,
			uint32_t signedness)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpTypeInt, wordCount);
			writeWords(idResult, width, signedness);
		}

		void OpTypeMatrix(
			IdResult idResult,
			IdRef columnType,
			uint32_t columnCount)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpTypeMatrix, wordCount);
			writeWords(idResult, columnType, columnCount);
		}

		void OpTypeNamedBarrier(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeNamedBarrier, wordCount);
			writeWords(idResult);
		}

		void OpTypeNodePayloadArrayAMDX(
			IdResult idResult,
			IdRef payloadType)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpTypeNodePayloadArrayAMDX, wordCount);
			writeWords(idResult, payloadType);
		}

		void OpTypeOpaque(
			IdResult idResult,
			const std::string& theNameOfTheOpaqueType)
		{
			uint16_t wordCount = 2;
			countOperandsWord(wordCount, theNameOfTheOpaqueType);

			writeWord(spv::Op::OpTypeOpaque, wordCount);
			writeWords(idResult, theNameOfTheOpaqueType);
		}

		void OpTypePipe(
			IdResult idResult,
			spv::AccessQualifier qualifier)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpTypePipe, wordCount);
			writeWords(idResult, qualifier);
		}

		void OpTypePipeStorage(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypePipeStorage, wordCount);
			writeWords(idResult);
		}

		void OpTypePointer(
			IdResult idResult,
			spv::StorageClass storageClass,
			IdRef type)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpTypePointer, wordCount);
			writeWords(idResult, storageClass, type);
		}

		void OpTypeQueue(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeQueue, wordCount);
			writeWords(idResult);
		}

		void OpTypeRayQueryKHR(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeRayQueryKHR, wordCount);
			writeWords(idResult);
		}

		void OpTypeReserveId(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeReserveId, wordCount);
			writeWords(idResult);
		}

		void OpTypeRuntimeArray(
			IdResult idResult,
			IdRef elementType)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpTypeRuntimeArray, wordCount);
			writeWords(idResult, elementType);
		}

		void OpTypeSampledImage(
			IdResult idResult,
			IdRef imageType)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpTypeSampledImage, wordCount);
			writeWords(idResult, imageType);
		}

		void OpTypeSampler(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeSampler, wordCount);
			writeWords(idResult);
		}

		void OpTypeStruct(
			IdResult idResult,
			const std::vector<IdRef>& memberTypes = {})
		{
			uint16_t wordCount = 2;
			countOperandsWord(wordCount, memberTypes);

			writeWord(spv::Op::OpTypeStruct, wordCount);
			writeWords(idResult, memberTypes);
		}

		void OpTypeStructContinuedINTEL(const std::vector<IdRef>& memberTypes = {})
		{
			uint16_t wordCount = 1;
			countOperandsWord(wordCount, memberTypes);

			writeWord(spv::Op::OpTypeStructContinuedINTEL, wordCount);
			writeWords(memberTypes);
		}

		void OpTypeTaskSequenceINTEL(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeTaskSequenceINTEL, wordCount);
			writeWords(idResult);
		}

		void OpTypeTensorLayoutNV(
			IdResult idResult,
			IdRef dim,
			IdRef clampMode)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpTypeTensorLayoutNV, wordCount);
			writeWords(idResult, dim, clampMode);
		}

		void OpTypeTensorViewNV(
			IdResult idResult,
			IdRef dim,
			IdRef hasDimensions,
			const std::vector<IdRef>& p = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, p);

			writeWord(spv::Op::OpTypeTensorViewNV, wordCount);
			writeWords(idResult, dim, hasDimensions, p);
		}

		void OpTypeUntypedPointerKHR(
			IdResult idResult,
			spv::StorageClass storageClass)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpTypeUntypedPointerKHR, wordCount);
			writeWords(idResult, storageClass);
		}

		void OpTypeVector(
			IdResult idResult,
			IdRef componentType,
			uint32_t componentCount)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpTypeVector, wordCount);
			writeWords(idResult, componentType, componentCount);
		}

		void OpTypeVmeImageINTEL(
			IdResult idResult,
			IdRef imageType)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpTypeVmeImageINTEL, wordCount);
			writeWords(idResult, imageType);
		}

		void OpTypeVoid(IdResult idResult)
		{
			uint16_t wordCount = 2;

			writeWord(spv::Op::OpTypeVoid, wordCount);
			writeWords(idResult);
		}

		void OpUAddSatINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpUAddSatINTEL, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpUAverageINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpUAverageINTEL, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpUAverageRoundedINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpUAverageRoundedINTEL, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpUConvert(
			IdResultType idResultType,
			IdResult idResult,
			IdRef unsignedValue)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpUConvert, wordCount);
			writeWords(idResultType, idResult, unsignedValue);
		}

		void OpUCountLeadingZerosINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpUCountLeadingZerosINTEL, wordCount);
			writeWords(idResultType, idResult, operand);
		}

		void OpUCountTrailingZerosINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpUCountTrailingZerosINTEL, wordCount);
			writeWords(idResultType, idResult, operand);
		}

		void OpUDiv(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpUDiv, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpUDot(
			IdResultType idResultType,
			IdResult idResult,
			IdRef vector1,
			IdRef vector2,
			std::optional<spv::PackedVectorFormat> packedVectorFormat = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, packedVectorFormat);

			writeWord(spv::Op::OpUDot, wordCount);
			writeWords(idResultType, idResult, vector1, vector2, packedVectorFormat);
		}

		void OpUDotAccSat(
			IdResultType idResultType,
			IdResult idResult,
			IdRef vector1,
			IdRef vector2,
			IdRef accumulator,
			std::optional<spv::PackedVectorFormat> packedVectorFormat = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, packedVectorFormat);

			writeWord(spv::Op::OpUDotAccSat, wordCount);
			writeWords(idResultType, idResult, vector1, vector2, accumulator, packedVectorFormat);
		}

		void OpUGreaterThan(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpUGreaterThan, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpUGreaterThanEqual(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpUGreaterThanEqual, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpULessThan(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpULessThan, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpULessThanEqual(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpULessThanEqual, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpUMod(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpUMod, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpUMul32x16INTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpUMul32x16INTEL, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpUMulExtended(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpUMulExtended, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpUSubSatINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef operand1,
			IdRef operand2)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpUSubSatINTEL, wordCount);
			writeWords(idResultType, idResult, operand1, operand2);
		}

		void OpUndef(
			IdResultType idResultType,
			IdResult idResult)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpUndef, wordCount);
			writeWords(idResultType, idResult);
		}

		void OpUnordered(
			IdResultType idResultType,
			IdResult idResult,
			IdRef x,
			IdRef y)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpUnordered, wordCount);
			writeWords(idResultType, idResult, x, y);
		}

		void OpUnreachable()
		{
			uint16_t wordCount = 1;

			writeWord(spv::Op::OpUnreachable, wordCount);
			writeWords();
		}

		void OpUntypedAccessChainKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef baseType,
			IdRef base,
			const std::vector<IdRef>& indexes = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, indexes);

			writeWord(spv::Op::OpUntypedAccessChainKHR, wordCount);
			writeWords(idResultType, idResult, baseType, base, indexes);
		}

		void OpUntypedArrayLengthKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef structure,
			IdRef pointer,
			uint32_t arrayMember)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpUntypedArrayLengthKHR, wordCount);
			writeWords(idResultType, idResult, structure, pointer, arrayMember);
		}

		void OpUntypedInBoundsAccessChainKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef baseType,
			IdRef base,
			const std::vector<IdRef>& indexes = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, indexes);

			writeWord(spv::Op::OpUntypedInBoundsAccessChainKHR, wordCount);
			writeWords(idResultType, idResult, baseType, base, indexes);
		}

		void OpUntypedInBoundsPtrAccessChainKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef baseType,
			IdRef base,
			IdRef element,
			const std::vector<IdRef>& indexes = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, indexes);

			writeWord(spv::Op::OpUntypedInBoundsPtrAccessChainKHR, wordCount);
			writeWords(idResultType, idResult, baseType, base, element, indexes);
		}

		void OpUntypedPrefetchKHR(
			IdRef pointerType,
			IdRef numBytes,
			std::optional<IdRef> RW = {},
			std::optional<IdRef> locality = {},
			std::optional<IdRef> cacheType = {})
		{
			uint16_t wordCount = 3;
			countOperandsWord(wordCount, RW, locality, cacheType);

			writeWord(spv::Op::OpUntypedPrefetchKHR, wordCount);
			writeWords(pointerType, numBytes, RW, locality, cacheType);
		}

		void OpUntypedPtrAccessChainKHR(
			IdResultType idResultType,
			IdResult idResult,
			IdRef baseType,
			IdRef base,
			IdRef element,
			const std::vector<IdRef>& indexes = {})
		{
			uint16_t wordCount = 6;
			countOperandsWord(wordCount, indexes);

			writeWord(spv::Op::OpUntypedPtrAccessChainKHR, wordCount);
			writeWords(idResultType, idResult, baseType, base, element, indexes);
		}

		void OpUntypedVariableKHR(
			IdResultType idResultType,
			IdResult idResult,
			spv::StorageClass storageClass,
			std::optional<IdRef> dataType = {},
			std::optional<IdRef> initializer = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, dataType, initializer);

			writeWord(spv::Op::OpUntypedVariableKHR, wordCount);
			writeWords(idResultType, idResult, storageClass, dataType, initializer);
		}

		void OpVariable(
			IdResultType idResultType,
			IdResult idResult,
			spv::StorageClass storageClass,
			std::optional<IdRef> initializer = {})
		{
			uint16_t wordCount = 4;
			countOperandsWord(wordCount, initializer);

			writeWord(spv::Op::OpVariable, wordCount);
			writeWords(idResultType, idResult, storageClass, initializer);
		}

		void OpVariableLengthArrayINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef lenght)
		{
			uint16_t wordCount = 4;

			writeWord(spv::Op::OpVariableLengthArrayINTEL, wordCount);
			writeWords(idResultType, idResult, lenght);
		}

		void OpVectorExtractDynamic(
			IdResultType idResultType,
			IdResult idResult,
			IdRef vector,
			IdRef index)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpVectorExtractDynamic, wordCount);
			writeWords(idResultType, idResult, vector, index);
		}

		void OpVectorInsertDynamic(
			IdResultType idResultType,
			IdResult idResult,
			IdRef vector,
			IdRef component,
			IdRef index)
		{
			uint16_t wordCount = 6;

			writeWord(spv::Op::OpVectorInsertDynamic, wordCount);
			writeWords(idResultType, idResult, vector, component, index);
		}

		void OpVectorShuffle(
			IdResultType idResultType,
			IdResult idResult,
			IdRef vector1,
			IdRef vector2,
			const std::vector<uint32_t>& components = {})
		{
			uint16_t wordCount = 5;
			countOperandsWord(wordCount, components);

			writeWord(spv::Op::OpVectorShuffle, wordCount);
			writeWords(idResultType, idResult, vector1, vector2, components);
		}

		void OpVectorTimesMatrix(
			IdResultType idResultType,
			IdResult idResult,
			IdRef vector,
			IdRef matrix)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpVectorTimesMatrix, wordCount);
			writeWords(idResultType, idResult, vector, matrix);
		}

		void OpVectorTimesScalar(
			IdResultType idResultType,
			IdResult idResult,
			IdRef vector,
			IdRef scalar)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpVectorTimesScalar, wordCount);
			writeWords(idResultType, idResult, vector, scalar);
		}

		void OpVmeImageINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef imageType,
			IdRef sampler)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpVmeImageINTEL, wordCount);
			writeWords(idResultType, idResult, imageType, sampler);
		}

		void OpWritePackedPrimitiveIndices4x8NV(
			IdRef indexOffset,
			IdRef packedIndices)
		{
			uint16_t wordCount = 3;

			writeWord(spv::Op::OpWritePackedPrimitiveIndices4x8NV, wordCount);
			writeWords(indexOffset, packedIndices);
		}

		void OpWritePipe(
			IdResultType idResultType,
			IdResult idResult,
			IdRef pipe,
			IdRef pointer,
			IdRef packetSize,
			IdRef packetAlignment)
		{
			uint16_t wordCount = 7;

			writeWord(spv::Op::OpWritePipe, wordCount);
			writeWords(idResultType, idResult, pipe, pointer, packetSize, packetAlignment);
		}

		void OpWritePipeBlockingINTEL(
			IdResultType idResultType,
			IdResult idResult,
			IdRef packetSize,
			IdRef packetAlignment)
		{
			uint16_t wordCount = 5;

			writeWord(spv::Op::OpWritePipeBlockingINTEL, wordCount);
			writeWords(idResultType, idResult, packetSize, packetAlignment);
		}
	};
} // namespace dynspv