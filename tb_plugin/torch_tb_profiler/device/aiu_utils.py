# -------------------------------------------------------------------------
# Copyright (c) IBM Corporation. All rights reserved.
# -------------------------------------------------------------------------

# pyre-unsafe
import json
from typing import Dict
import os

from .. import io, utils

logger = utils.get_logger()

def run_acelyzer(trace_path: str, trace_json: Dict, json_reencode: bool):
    acelyzer_opt = os.environ.get('ACELYZER_OPT', '')
    if acelyzer_opt != 'disable':
        try:
            from aiu_trace_analyzer.core.acelyzer import Acelyzer
            logger.info("Running Acelyzer for AIU trace preprocessing. "
                        "Disable by setting ACELYZER_OPT=disable.")
            
            # default acelyzer args
            acelyzer_args = {
                "--input": "api://jsonbuffer",
                "--disable_file": "",
                "--tb": "",
                "--overlap": "shift"
            }

            # if there's a compiler log file in the same directory as the trace file, pass it to acelyzer
            compiler_log_path = io.join(os.path.dirname(trace_path), "compiler-log.txt")
            if io.exists(compiler_log_path):
                logger.info(f"Found compiler log file at {compiler_log_path}, passing it to Acelyzer.")
                compiler_log_abs_path = io.abspath(compiler_log_path)
                acelyzer_args["--compiler_log"] = compiler_log_abs_path
            else:
                logger.info(f"Compiler log file not found. Provide one at {compiler_log_path} to see PT utilization.")

            # override/add args provided in env var ACELYZER_OPT
            # ACELYZER_OPT is a comma-separated list of key=value or key (for flags)
            # e.g. ACELYZER_OPT="tb,compiler_log=log.txt"
            opts = acelyzer_opt.split(',')
            if opts == ['']:
                opts = []
            else:
                logger.info(f"Overriding Acelyzer options from env var ACELYZER_OPT={acelyzer_opt}.")
            for opt in opts:
                kv = opt.split('=')
                if len(kv) == 2:
                    # allow --disable_file and --tb to be disabled
                    if kv[0] == "disable_file" and kv[1].lower() == "false":
                        if "--disable_file" in acelyzer_args:
                            del acelyzer_args["--disable_file"]
                        continue
                    if kv[0] == "tb" and kv[1].lower() == "false":
                        if "--tb" in acelyzer_args:
                            del acelyzer_args["--tb"]
                        continue

                    # ensure short form args override default long form args
                    if kv[0] == "c":
                        kv[0] = "compiler_log"
                    if kv[0] == "i":
                        kv[0] = "input"
                    if kv[0] == "P":
                        kv[0] = "profile"
                    if kv[0] == "O":
                        kv[0] = "overlap"

                    if len(kv[0]) == 1:
                        acelyzer_args["-" + kv[0]] = kv[1]
                    else:
                        acelyzer_args["--" + kv[0]] = kv[1]
                else:
                    if len(kv[0]) == 1:
                        acelyzer_args["-" + kv[0]] = ""
                    else:
                        acelyzer_args["--" + kv[0]] = ""
            
            args_list = []
            for k, v in acelyzer_args.items():
                args_list.append(k)
                if v != "":
                    args_list.append(v)

            trace_data = json.dumps(trace_json).encode()
            logger.debug("Running Acelyzer with in_args=%s", args_list)
            ace = Acelyzer(args_list, in_data=trace_data)
            ace.run()
            processed_data = ace.get_output_data()
            trace_json = json.loads(processed_data)
            json_reencode = True
        except ImportError:
            logger.warning("Module aiu_trace_analyzer not found, install with "
                            "`pip install aiu-trace-analyzer` or set ACELYZER_OPT=disable "
                            "to silence this warning.")
    return trace_json, json_reencode