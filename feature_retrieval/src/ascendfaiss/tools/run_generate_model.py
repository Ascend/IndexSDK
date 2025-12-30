#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
import os
import signal
import time
import logging
import argparse
import subprocess
from xml.dom.minidom import parse
import common as utils
from common import generate_work_dir

logging.basicConfig(level=logging.NOTSET, format="")

_SUPPORTED_MODES = [
    "Flat", "SQ8", "IVFSQ8", "INT8", "ALL"]


def arg_parse():
    """
    Parse arguements to the operator model
    """
    parser = argparse.ArgumentParser(
        description="generate linear_transform operator model")

    utils.op_common_parse(parser, "-m", "mode", "ALL", str, 
                          "Generate mode, include combination of {}".format(", ".join(_SUPPORTED_MODES)))
    utils.op_common_parse(parser, "-p", "pipeline", "true", str, "pipeline, true or false. default true")
    utils.op_common_parse(parser, "-t", "npu_type", "310", str, "NPU type, 310 or 310P. 310 by default")
    utils.op_common_parse(parser, "-pool", "pool_size", 32, int, "Number of pool_size")
    return parser.parse_args()


def get_field_value(node, field):
    value = node.getElementsByTagName(field)[0]
    return int(value.childNodes[0].data)


def get_attribute_value(node, attribute):
    return int(node.getAttribute(attribute))


class GenerateModel:
    def __init__(self, process_id, parameter, para_mode, model_id, pipeline_flag, npu_type):
        self.err_code = 0
        self.subprocess = None
        self.cmd = None
        self.process_id = process_id
        self.parameter = parameter
        self.para_mode = para_mode
        self.model_id = model_id
        self.pipeline_flag = pipeline_flag
        self.npu_type = npu_type
        self.pool_size = 1
        return

    def run_command(self, arguments, adapted_to_310p=True):
        if "-p" in arguments:
            logging.info("error! option -p has been occupied")
            return
        arguments.append("-p")
        arguments.append("%d" % self.process_id)
        arguments += ["-t", self.npu_type]

        # Set cores number for 310P
        if adapted_to_310p and self.npu_type == "310P":
            arguments += ["--cores", "8"]
        

        cmd = " ".join(arguments)
        self.cmd = cmd
        logging.info("generate model of ID %s, cmd: %s\r\n", self.model_id, cmd)

        if not self.pipeline_flag:
            self.err_code = subprocess.call(arguments, shell=False)  # Serial processing
        else:
            self.subprocess = subprocess.Popen(arguments, shell=False)  # Parallel processing
        return

    def run_flat(self):
        dim = get_field_value(self.parameter, "dim")

        arguments = [
            "python3", "flat_generate_model.py",
            "-d", "%d" % dim,
            "-pool", "%d" % self.pool_size
        ]
        self.run_command(arguments)
        return

    def run_sq8(self):
        dim = get_field_value(self.parameter, "dim")

        arguments = [
            "python3", "sq8_generate_model.py",
            "-d", "%d" % dim,
            "-pool", "%d" % self.pool_size
        ]
        self.run_command(arguments)
        return

    def run_ivf_sq8(self):
        dim = get_field_value(self.parameter, "dim")
        coarse_centroid_num_ = get_field_value(self.parameter, "nlist")

        arguments = [
            "python3", "ivfsq8_generate_model.py",
            "-d", "%d" % dim,
            "-c", "%d" % coarse_centroid_num_,
            "-pool", "%d" % self.pool_size
        ]
        self.run_command(arguments)
        return

    def run_int8_flat(self):
        dim = get_field_value(self.parameter, "dim")

        arguments = [
            "python3", "int8flat_generate_model.py",
            "-d", "%d" % dim,
            "-pool", "%d" % self.pool_size
        ]
        self.run_command(arguments)
        return

    def run_aicpu(self):
        arguments = [
            'python3', 'aicpu_generate_model.py'
        ]
        self.run_command(arguments)
        return

    def generate_offline_model(self, pool_size):
        self.pool_size = pool_size
        if self.para_mode == "Flat":
            self.run_flat()
        elif self.para_mode == "SQ8":
            self.run_sq8()
        elif self.para_mode == "IVFSQ8":
            self.run_ivf_sq8()
        elif self.para_mode == "INT8":
            self.run_int8_flat()
        elif self.para_mode == 'AICPU':
            self.run_aicpu()
        return

    def get_err_code(self):
        return self.err_code

    def get_subprocess(self):
        return self.subprocess

    def get_cmd(self):
        return self.cmd


def print_process_status(process):
    logging.info("\r\n%s", "-" * 100)
    logging.info("%-10s%-10s%-12s%-15s%s", "model_id", "model", "status", "running time", "cmd")
    logging.info("%s", "-" * 100)
    for model_id in process:
        mode, _, status, running_time, cmd = process[model_id]
        running_time = int(running_time) if ((status == "success") or (status == "failed")) else "-"
        logging.info("%-10s%-10s%-12s%-15s%s", model_id, mode, status, running_time, cmd)
    logging.info("%s\r\n", "-" * 100)
    return


def wait_to_finish(process):
    print_timer = 0
    print_process_status(process)

    while True:
        time.sleep(1)

        finished = True
        for model_id, proc in process.items():
            mode, sub_process, status, time_, cmd = proc
            if not (sub_process and (status == "processing")):
                continue
            finished = False
            if sub_process.poll() is None:
                continue

            running_time = time.time() - time_

            if sub_process.returncode == 0 and running_time > 3:
                process[model_id] = ([mode, sub_process, "success", running_time, cmd])
            else:
                logging.info("ERROR: generate model of ID %s failed", model_id)
                process[model_id] = ([mode, sub_process, "failed", running_time, cmd])
            print_process_status(process)

        if finished:
            break

        print_timer += 1
        if print_timer >= 30:
            print_timer = 0
            print_process_status(process)

    return process


def count_process(parameters):
    process = {}
    for parameter in parameters:
        if parameter.hasAttribute("ID"):
            model_id = get_attribute_value(parameter, "ID")
            mode_ = parameter.getElementsByTagName("mode")[0]
            para_mode = str(mode_.childNodes[0].data)
            process[model_id] = ([para_mode, None, "waiting", "None", "None"])
    return process


def generate_oms(parameters, arg_mode, pipeline_flag, npu_type, pool_size):
    modes = arg_mode.split(",")
    process = count_process(parameters)
    process_id = 0

    for parameter in parameters:
        if not parameter.hasAttribute("ID"):
            continue
        model_id = get_attribute_value(parameter, "ID")
        mode_ = parameter.getElementsByTagName("mode")[0]
        para_mode = str(mode_.childNodes[0].data)

        if arg_mode == "ALL" or para_mode in modes:
            generate_model = GenerateModel(process_id, parameter, para_mode, model_id, pipeline_flag, npu_type)
            process_id += 1

            if pipeline_flag:
                generate_model.generate_offline_model(pool_size)
                process[model_id] = ([para_mode, generate_model.get_subprocess(),
                                      "processing", time.time(), generate_model.get_cmd()])
            else:
                process[model_id] = ([para_mode, None, "processing", None, generate_model.get_cmd()])
                print_process_status(process)

                start_time_ = time.time()
                generate_model.generate_offline_model(pool_size)
                running_time = time.time() - start_time_

                if generate_model.get_err_code() == 0:
                    process[model_id] = ([para_mode, None, "success", running_time, generate_model.get_cmd()])
                else:
                    process[model_id] = ([para_mode, None, "failed", running_time, generate_model.get_cmd()])
                    logging.info("ERROR: generate model of ID %s failed", model_id)
                print_process_status(process)
        else:
            continue

    try:
        if pipeline_flag:
            wait_to_finish(process)
    except KeyboardInterrupt:
        for _, proc in process.items():
            mode, sub_process, status, time_, cmd = proc
            pid = sub_process.pid
            try:
                os.killpg(pid, signal.SIGKILL)
            except KeyError:
                logging.error(" pid %d has been killed", pid)
        return {}

    return process


def pipeline_generate_oms(parameters, max_pool, arg_mode, npu_type):
    # Split operator parameters by category
    split_paras_dict = {}
    for para in parameters:
        mode_ = para.getElementsByTagName("mode")[0]
        para_mode = str(mode_.childNodes[0].data)
        if para_mode not in split_paras_dict:
            split_paras_dict[para_mode] = []
        try:
            split_paras_dict[para_mode].append(para)
        except KeyError as error:
            raise ValueError(f"key {para_mode} not in split_paras_dict ") from error

    all_process = {}
    # Generate operators by category loop
    for para_mode in split_paras_dict:
        start = time.time()

        try:
            parameters = split_paras_dict[para_mode]
        except KeyError as error:
            raise ValueError(f"key {para_mode} not in split_paras_dict ") from error

        pool_size = max_pool // len(parameters)
        pipeline_flag = True if pool_size > 0 else False
        pool_size = pool_size if pipeline_flag else 32
        one_type_process = generate_oms(parameters, arg_mode, pipeline_flag, npu_type, pool_size)
        if len(one_type_process) == 0:
            break
        all_process.update(one_type_process)
        logging.info("%s generate model time: %.3f seconds ", para_mode, time.time() - start)

    print_process_status(all_process)


def is_config_file_valid(config_path):
    invalid_fields = ["DOCTYPE", "ENTITY", "SCHEMA", "XSI", "XMLNS"]

    with open(config_path, 'r') as xml_file:
        file_context = xml_file.read().upper()
        for field in invalid_fields:
            if field in file_context:
                return False
    return True


def main():
    start_time = time.time()
    args = arg_parse()
    arg_mode = args.mode
    npu_type = args.npu_type
    pool_size = args.pool_size
    max_pool = 32
    max_para_table_size = 20480

    logging.info("\r\nModels of %s would be generated\r\n", npu_type)

    if npu_type not in ["310", "310P", "910B1", "910B2", "910B3", "910B4"]:
        raise RuntimeError("NPU type only support 310 / 310P / 910B1 / 910B2 / 910B3 / 910B4 now!")

    generate_work_dir(work_dir=".")

    modes = arg_mode.split(",")
    for mode in modes:
        if mode not in _SUPPORTED_MODES:
            raise RuntimeError("generate mode only support {}".format(", ".join(_SUPPORTED_MODES)))
    config_name = "./para_table.xml"
    if os.path.getsize(config_name) > max_para_table_size:
        raise RuntimeError("para_table.xml file size exceeds the limit")
    if not is_config_file_valid(config_name):
        raise RuntimeError("para_table.xml contains illegal fields[DOCTYPE, ENTITY, SCHEMA, XSI, XMLNS]")

    dom_tree = parse(config_name)
    root = dom_tree.documentElement
    parameters = root.getElementsByTagName("parameter")

    if args.pipeline.lower() == "true":
        pipeline_flag = True
        logging.info("\r\npipeline_flag = %s\r\n", pipeline_flag)
        pipeline_generate_oms(parameters, max_pool, arg_mode, npu_type)
    else:
        pipeline_flag = False
        logging.info("\r\npipeline_flag = %s\r\n", pipeline_flag)
        if pool_size > max_pool or pool_size < 1:
            raise ValueError("when pipeline is False, Input variable pool_size should be [1, 32]")
        generate_oms(parameters, arg_mode, pipeline_flag, npu_type, pool_size)

    logging.info("generate model time duration: %.3f seconds", time.time() - start_time)


if __name__ == "__main__":
    main()
