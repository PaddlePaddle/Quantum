# !/usr/bin/env python3
# Copyright (c) 2021 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
Quantum portfolio optimization.
"""
import os 
import sys 
from typing import Dict
import logging
import argparse
import toml
import datetime
import pandas as pd

from paddle_quantum.finance.qpo import portfolio_combination_optimization
from paddle_quantum.finance import DataSimulator


def main(args):
    # logger configure
    log_path = args.logger
    logger = logging.Logger(name='logger_qpo')
    logger_file_handler = logging.FileHandler(log_path)
    logger_file_handler.setFormatter(logging.Formatter(r'%(levelname)s  %(asctime)s  %(message)s'))
    logger_file_handler.setLevel(logging.INFO)
    logger.addHandler(logger_file_handler)
    logger.warning("------------------- Process starts -------------------")

    # data preparation
    parsed_configs: Dict = toml.load(args.config)
    num_asset = parsed_configs["stock_para"]["num_asset"]

    if parsed_configs['stock'] == 'demo':
        stock_file_path = os.path.join(this_file_path, './demo_stock.csv')
        stocks_name = [("STOCK%s" % i) for i in range(num_asset)]
        source_data = pd.read_csv(stock_file_path)
        processed_data = [source_data['closePrice'+str(i)].tolist() for i in range(num_asset)]
        data = DataSimulator(stocks_name)
        data.set_data(processed_data)
        logger.warning(f"******************* {num_asset} stocks processed *******************")
        
    elif parsed_configs['stock'] == 'random':
        stocks_name = [("STOCK%s" % i) for i in range(num_asset)]
        data = DataSimulator(stocks=stocks_name, start=datetime.datetime(
            *parsed_configs['random_data']['start_time']), end=datetime.datetime(*parsed_configs['random_data']['end_time']))
        data.randomly_generate()
        logger.warning(f"******************* {num_asset} stocks randomly generated *******************")

    elif parsed_configs['stock'] == 'custom':
        stock_file_path = parsed_configs["custom_data_path"]
        stocks_name = [("STOCK%s" % i) for i in range(num_asset)]
        source_data = pd.read_csv(stock_file_path)
        processed_data = [source_data['closePrice'+str(i)].tolist() for i in range(num_asset)]
        data = DataSimulator(stocks_name)
        data.set_data(processed_data)
        logger.warning(f"******************* {num_asset} stocks processed *******************")

    # load model parameters
    risk_weight = parsed_configs["stock_para"]["risk_weight"]
    budget = parsed_configs["stock_para"]["budget"]
    penalty = parsed_configs["stock_para"]["penalty"]
    circuit_depth = parsed_configs["train_para"]["circuit_depth"]
    iters = parsed_configs["train_para"]["iterations"]
    lr = parsed_configs["train_para"]["learning_rate"]

    # optimization
    logger.warning("******************* Train starts *******************")
    invest = portfolio_combination_optimization(num_asset, data, iters, lr, risk_weight, budget,
                                       penalty, circuit=circuit_depth, logger=logger, compare=True)
    logger.warning("******************* Train ends *******************")
    logger.warning(f"******************* Output is {invest}  *******************")
    logger.warning("------------------- Process ends -------------------")


if __name__ == "__main__":
    this_file_path = sys.path[0]
    parser = argparse.ArgumentParser(description="Quantum chemistry task with paddle quantum.")
    parser.add_argument(
        "--config", default=os.path.join(this_file_path, './config.toml'), type=str, help="The path of toml format config file.")
    parser.add_argument(
        "--logger", default=os.path.join(this_file_path, './qpo_log.log'), type=str, help="The path of log file saved.")
    main(parser.parse_args())





