# SPDX-License-Identifier: MPL-2.0

import json
import subprocess
import re
from collections import Counter


def run_clang_format(file_path: str):
    subprocess.run(["clang-format", "-i", file_path], check=True)


reserved_cpp_keywords = ["asm", "auto", "break", "case", "catch", "char", "class", "const", "continue", "default", "delete", "do", "double", "else", "enum", "extern", "float", "for", "friend", "goto", "if", "inline", "int", "long", "new",
                         "operator", "private", "protected", "public", "register", "return", "short", "signed", "sizeof", "static", "struct", "switch", "template", "this", "throw", "try", "typedef", "union", "unsigned", "virtual", "void", "volatile", "while",]

with open("spirv.core.grammar.json", "r") as file:
    spirv_grammar = json.load(file)

operands = spirv_grammar["operand_kinds"]
operands = map(lambda x: (x["kind"], x), operands)
operands = dict(operands)


def decapitalize(s: str) -> str:
    if s.isupper():
        return s
    return s[:1].lower() + s[1:]


def capitalize(s: str) -> str:
    return s[:1].upper() + s[1:]


literal_dict = {
    'LiteralInteger': 'uint32_t',
    'LiteralString': 'std::string',
    'LiteralFloat': 'float',
    'LiteralContextDependentNumber': 'spvConstant auto',
    'LiteralExtInstInteger': 'uint32_t',
    'LiteralSpecConstantOpInteger': 'uint32_t',
}


def get_base_cpp_type(operand_kind: str) -> str:
    operand_info = operands[operand_kind]
    operand_category = operand_info["category"]

    match operand_category:
        case "Id": cpp_type = operand_kind
        case "Literal": cpp_type = literal_dict[operand_kind]
        case "ValueEnum": cpp_type = f"spv::{operand_kind}"
        case "BitEnum": cpp_type = f"spv::{operand_kind}Mask"
        case "Composite":
            bases = operand_info["bases"]
            bases = map(lambda x: operands[x]["kind"], bases)
            bases = map(get_base_cpp_type, bases)
            bases = ", ".join(bases)
            cpp_type = f"std::tuple<{bases}>"
            pass
        case _: raise NotImplementedError(f"Unexpected operand category", operand_category)

    return cpp_type


def get_cpp_type(operand: dict) -> str:
    operand_kind = operand["kind"]

    cpp_type = get_base_cpp_type(operand_kind)

    quantifier = None
    if "quantifier" in operand:
        quantifier = operand["quantifier"]

    match quantifier:
        case "?": cpp_type = f"std::optional<{cpp_type}>"
        case "*": cpp_type = f"std::vector<{cpp_type}>"
        case None: pass
        case _: raise NotImplementedError(f"Unexpected operand quantifier", quantifier)

    return cpp_type


def get_cpp_param_name(operand: dict) -> str:
    if "name" not in operand:
        return decapitalize(operand["kind"])

    name = operand["name"]

    if "\n" in name:
        name = re.search(r"'([\w ]*)', ", name).group(1)
        name = re.sub(r" \d", "", name)
        name = f"{name}s"
    if re.match(r"'[A-Za-z_, ]+ \.\.\.'", name) is not None:
        name = re.findall(r"([\w]+)", name)
        name = " ".join(name)
        name = f"{name}s"
    if "..." in name:
        name = re.findall(r"([\w]+)", name)
        temp_name = map(lambda x: re.sub(r"\d", "", x), name)
        temp_name = map(str.strip, temp_name)
        temp_name = list(set(temp_name))
        assert len(temp_name) == 1
        name = f"{temp_name[0]}s"
    name = name.split(" ")
    name = map(capitalize, name)
    name = "".join(name)
    name = re.sub(r"[^\w]", "", name)

    name = decapitalize(name)

    if name in reserved_cpp_keywords:
        name = f"_{name}"

    return name


def get_cpp_param(operand: dict) -> dict:
    return {
        "type": get_cpp_type(operand),
        "name": get_cpp_param_name(operand),
        "default_value": ""
    }


def get_word_count_code(cpp_params: list[dict]) -> str:
    word_count = 1

    def is_simple_type(cpp_type: str) -> bool:
        if cpp_type == "std::string" or cpp_type == "spvConstant auto":
            return False
        if cpp_type.startswith("std::optional"):
            return False
        if cpp_type.startswith("std::vector"):
            return False
        if cpp_type.startswith("std::tuple"):
            return False
        return True

    simple_types = map(lambda x: x["type"], cpp_params)
    simple_types = filter(is_simple_type, simple_types)
    word_count = word_count+len(list(simple_types))

    non_simple_types = filter(
        lambda x: not is_simple_type(x["type"]), cpp_params)
    non_simple_types = map(lambda x: x['name'], non_simple_types)
    non_simple_types_code = ', '.join(non_simple_types)
    if len(non_simple_types_code) > 0:
        non_simple_types_code = f"countOperandsWord(wordCount,{non_simple_types_code});"

    return f"""uint16_t wordCount = {word_count};
    {non_simple_types_code}"""


def get_instruction_write_code(cpp_params: list[dict]) -> str:
    write_code = map(lambda x: x["name"], cpp_params)
    write_code = ", ".join(write_code)
    write_code = f"writeWords({write_code});"
    return write_code


def get_instruction_code(instruction: dict) -> str:
    cpp_params = []

    if "operands" in instruction:
        cpp_params = instruction["operands"]

    cpp_params = map(get_cpp_param, cpp_params)
    cpp_params = list(cpp_params)

    param_names_counter = map(lambda x: x["name"], cpp_params)
    param_names_counter = Counter(param_names_counter)
    for param_name, count in param_names_counter.items():
        if count == 1:
            continue
        param_nr = 1

        for cpp_param in cpp_params:
            if cpp_param["name"] == param_name:
                cpp_param["name"] = f"{param_name}{param_nr}"
                param_nr = param_nr+1

    for instruction_operand in reversed(cpp_params):
        instruction_type = instruction_operand["type"]
        if not (instruction_type.startswith("std::optional") or instruction_type.startswith("std::vector")):
            break
        instruction_operand["default_value"] = " = {}"

    def get_param_def(function_param: dict) -> str:
        x = function_param
        cpp_type = function_param['type']
        if cpp_type == "std::string":
            cpp_type = "const std::string&"
        if cpp_type.startswith("std::vector"):
            cpp_type = re.sub(r"std::vector<(.*)>",
                              r"const std::vector<\1>&", cpp_type)
        if cpp_type.startswith("std::tuple"):
            cpp_type = re.sub(r"std::tuple<(.*)>",
                              r"const std::tuple<\1>&", cpp_type)

        return f"{cpp_type} {function_param['name']}{function_param['default_value']}"
    opname = instruction["opname"]
    function_params = cpp_params
    function_params = map(get_param_def,  function_params)
    function_params = ",\n".join(function_params)
    if len(cpp_params) > 1:
        function_params = "\n"+function_params

    return f"""
    void {opname}({function_params})
    {{
    {get_word_count_code(cpp_params)}

    writeWord(spv::Op::{opname}, wordCount);
    {get_instruction_write_code(cpp_params)}
    }}"""


def get_spv_ids_types():
    id_operands = spirv_grammar["operand_kinds"]
    id_operands = filter(lambda x: x["category"] == "Id", id_operands)
    id_operands = sorted(id_operands, key=lambda x: x["kind"])
    id_operands = map(lambda x: f"using {x['kind']} = spv::Id;", id_operands)
    id_operands = "\n".join(id_operands)
    return id_operands


lib_path = "include/dynspv.hpp"

with open("dynspv.hpp_template", "r") as file:
    lib_core_content = file.read()

with open(lib_path, "w") as file:
    instructions = spirv_grammar["instructions"]
    instructions = sorted(instructions, key=lambda x: x["opname"])
    instructions = map(get_instruction_code, instructions)
    instructions = "\n".join(instructions)

    lib_core_content = lib_core_content.replace(
        "#generated_code", instructions)
    lib_core_content = lib_core_content.replace(
        "#generated_spv_id_types", get_spv_ids_types())
    file.write(lib_core_content.replace("#generated_code", instructions))

run_clang_format(lib_path)
