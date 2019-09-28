# DeepRL-TradeBot

## General

Research project on applying deep reinforcement learning to perform financial market predictions. Independent research conducted by Mao L. and Songling L., guided by professor Sicun Gao, UC San Diego Jacobs School of Engineering. 

## Dataset

Real-time and historical data on stocks, provided by Alpha Vintage. 

## Method

Deep reinforcement learning

## Installation

First start a python (Python3.7 & Pip3) virtual environment as following:
```
python3 -m venv venv
source venv/bin/activate
```
Then install all neccessary packages using:
```
pip install -r requirement
```
Put your own Alpha Vantage API key in the `.env` file under `/config`. 

## Reference

- [Market Making via Reinforcement Learning](https://arxiv.org/abs/1804.04216)
- [Constrained Policy Optimization](https://arxiv.org/abs/1705.10528)
