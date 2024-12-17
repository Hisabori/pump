# 정리된 import 문
import ccxt
import pandas as pd
import time
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM


import numpy as np
import pandas_ta as ta  # pandas_ta 추가

def get_exchange_data(exchange_name):
    """
    거래소 인스턴스 생성
    """
    exchange_class = getattr(ccxt, exchange_name)
    return exchange_class({"rateLimit": 1200, "enableRateLimit": True})

def fetch_ohlcv(exchange, symbol, timeframe='5m', limit=50):
    """
    OHLCV 데이터를 가져옵니다.
    :param exchange: 거래소 인스턴스
    :param symbol: 거래 심볼
    :param timeframe: 시간 프레임
    :param limit: 데이터 길이
    :return: pandas DataFrame
    """
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def apply_technical_indicators(df):
    """
    여러 기술 지표를 계산하여 추가합니다.
    :param df: OHLCV 데이터프레임
    :return: 업데이트된 DataFrame
    """
    # 이동평균선 (MA)
    df['MA50'] = ta.sma(df['close'], length=50)
    df['MA200'] = ta.sma(df['close'], length=200)

    # RSI (Relative Strength Index)
    df['RSI'] = ta.rsi(df['close'], length=14)

    # 볼린저 밴드 (Bollinger Bands)
    bbands = ta.bbands(df['close'], length=20, std=2)
    df = pd.concat([df, bbands], axis=1)

    # MACD (Moving Average Convergence Divergence)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)

    # Fibonacci Retracement (단순 계산 예시)
    df['fib_high'] = df['high'].max()
    df['fib_low'] = df['low'].min()
    df['fib_0.236'] = df['fib_high'] - (df['fib_high'] - df['fib_low']) * 0.236
    df['fib_0.382'] = df['fib_high'] - (df['fib_high'] - df['fib_low']) * 0.382
    df['fib_0.618'] = df['fib_high'] - (df['fib_high'] - df['fib_low']) * 0.618

    # ATR (Average True Range)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    # Supertrend
    supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=7, multiplier=3)
    df = pd.concat([df, supertrend], axis=1)

    return df

def analyze_pump_and_dump(df):
    """
    급등, 급락 직전 분석 함수
    :param df: 기술 지표가 적용된 OHLCV 데이터프레임
    :return: 분석 결과
    """
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    pump_10min = ((last_row['close'] - df['close'].iloc[-3]) / df['close'].iloc[-3]) * 100
    pump_1hr = ((last_row['close'] - df['close'].iloc[-12]) / df['close'].iloc[-12]) * 100
    dump_signal = last_row['RSI'] > 70 and last_row['close'] < prev_row['close']

    report = []
    if pump_10min > 5:  # 10분 내 급등 5% 이상
        report.append(f"10분 내 급등 감지: 상승률 {pump_10min:.2f}%")
    if pump_1hr > 10:  # 1시간 내 급등 10% 이상
        report.append(f"1시간 내 급등 감지: 상승률 {pump_1hr:.2f}%")
    if dump_signal:  # RSI 70 이상 + 하락 신호
        report.append("급락 직전 신호 감지: RSI 과매수 + 하락 시작")

    return report

def calculate_trade_levels(df):
    """
    진입가, 청산가, 손절가 계산
    :param df: 기술 지표가 적용된 데이터프레임
    :return: 진입가, 청산가, 손절가
    """
    last_row = df.iloc[-1]
    entry_price = last_row['close']
    take_profit = entry_price * 1.05  # 5% 수익 목표
    stop_loss = entry_price * 0.97  # 3% 손절 목표
    return entry_price, take_profit, stop_loss

def find_combined_signals(exchange, symbol):
    """
    다양한 지표를 종합하여 매수 및 매도 시점을 감지합니다.
    :param exchange: 거래소 인스턴스
    :param symbol: 거래 심볼
    :return: 신호 메시지 및 급등/급락 분석
    """
    df = fetch_ohlcv(exchange, symbol)
    df = apply_technical_indicators(df)
    last_row = df.iloc[-1]

    entry_price, take_profit, stop_loss = calculate_trade_levels(df)

    # 매수 조건: RSI가 30 이하, MACD가 시그널 위로 교차, 종가가 볼밴 하단 근처
    buy_signal = (
            last_row['RSI'] <= 30 and
            last_row['MACD_12_26_9'] > last_row['MACDs_12_26_9'] and
            last_row['close'] <= last_row['BBL_20_2.0']
    )

    # 매도 조건: RSI가 70 이상, MACD가 시그널 아래로 교차, 종가가 볼밴 상단 근처
    sell_signal = (
            last_row['RSI'] >= 70 and
            last_row['MACD_12_26_9'] < last_row['MACDs_12_26_9'] and
            last_row['close'] >= last_row['BBU_20_2.0']
    )

    analysis = analyze_pump_and_dump(df)

    if buy_signal:
        return f"{symbol}: 진입가 {entry_price:.2f}, 청산가 {take_profit:.2f}, 손절가 {stop_loss:.2f}", analysis
    elif sell_signal:
        return f"{symbol}: 진입가 {entry_price:.2f}, 청산가 {take_profit:.2f}, 손절가 {stop_loss:.2f}", analysis
    else:
        return None, analysis

def main():
    """
    메인 함수: 실시간 데이터 수집 및 종합 신호 분석
    """
    exchange_names = ['binance']  # 사용할 거래소 목록

    while True:
        print("\n=== 종합 신호 분석 및 리포트 ===")
        for exchange_name in exchange_names:
            print(f"\n[{exchange_name.upper()} 거래소]")
            try:
                exchange = get_exchange_data(exchange_name)
                markets = exchange.load_markets()
                usdt_pairs = [symbol for symbol in markets if symbol.endswith("/USDT")]

                alerts = []
                for symbol in usdt_pairs:  # 모든 USDT 페어 분석
                    signal, analysis = find_combined_signals(exchange, symbol)
                    if signal:
                        alerts.append(signal)
                        for note in analysis:
                            print(f"  - {note}")

                # 최상단에 알림 출력
                if alerts:
                    print("\n=== 특징 코인 알림 ===")
                    for alert in alerts:
                        print(alert)
            except Exception as e:
                print(f"오류 발생: {e}")
        time.sleep(60)  # 1분마다 데이터 갱신

if __name__ == "__main__":
    main()
