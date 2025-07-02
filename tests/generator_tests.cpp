#include <gtest/gtest.h>

#include <spirv-tools/libspirv.hpp>
#include <dynspv.hpp>
#include <format>
#include <sstream>

TEST(GeneratorTests, GenerateValidBasicShader)
{
	class GeneratorTest : public dynspv::ModuleGenerator
	{
	  public:
		GeneratorTest()
		{
			writeHeader(0x010000);
			OpCapability(spv::Capability::CapabilityShader);
			OpExtInstImport(nextId(), "GLSL.std.450");
			OpMemoryModel(spv::AddressingModel::AddressingModelLogical, spv::MemoryModel::MemoryModelGLSL450);
			auto mainId = nextId();
			OpEntryPoint(spv::ExecutionModel::ExecutionModelVertex, mainId, "main");
			OpSource(spv::SourceLanguage::SourceLanguageGLSL, 450);
			auto voidTypeId = nextId();
			OpName(mainId, "main");
			OpTypeVoid(voidTypeId);
			auto voidFunctionTypeId = nextId();
			OpTypeFunction(voidFunctionTypeId, voidTypeId);
			OpFunction(voidTypeId, mainId, spv::FunctionControlMask::FunctionControlMaskNone, voidFunctionTypeId);
			OpLabel(nextId());
			OpReturn();
			OpFunctionEnd();

			updateBound(getBound());
		}
	};

	GeneratorTest generator{};
	spvtools::SpirvTools spvTools{SPV_ENV_UNIVERSAL_1_6};
	std::stringstream errors;
	spvTools.SetMessageConsumer(
		[&errors](
			spv_message_level_t level,
			const char* source,
			const spv_position_t& position,
			const char* message) {
			errors << std::format("Source: {}; Message: {}\n", source, message);
		});
	bool isValid = spvTools.Validate(generator.getCode());
	ASSERT_TRUE(isValid) << errors.str();
}
