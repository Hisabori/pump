import ccxt
import pandas as pd
import time
import sqlite3
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
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

def apply_supertrend(df, length=7, multiplier=3):
    """
    슈퍼트렌드 지표 계산
    :param df: OHLCV 데이터프레임
    :param length: ATR 길이
    :param multiplier: 배수
    :return: 업데이트된 DataFrame
    """
    supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=length, multiplier=multiplier)
    df = pd.concat([df, supertrend], axis=1)
    return df

def find_supertrend_support(exchange, symbol):
    """
    슈퍼트렌드 기준 저점 감지 및 매수 표시
    :param exchange: 거래소 인스턴스
    :param symbol: 거래 심볼
    :return: 저점 상태 메시지 및 매수 시점
    """
    df = fetch_ohlcv(exchange, symbol)
    df = apply_supertrend(df)
    last_row = df.iloc[-1]

    if last_row['SUPERTd_7_3.0'] == 1:
        buy_price = last_row['close']
        return f"매수 시점 감지: {symbol}, 현재 가격 {buy_price:.2f}, 지지선 유지"
    else:
        return f"매수 대기: {symbol}, 현재 가격 {last_row['close']:.2f}"

def prepare_data(tickers):
    """
    딥러닝 모델용 데이터 준비
    :param tickers: 티커 데이터
    :return: numpy array (입력 데이터)
    """
    data = []
    for symbol, ticker in tickers.items():
        try:
            if ticker['last'] and ticker['open'] and symbol.endswith("/USDT"):
                percentage_change = ((ticker['last'] - ticker['open']) / ticker['open']) * 100
                data.append([ticker['last'], ticker['open'], percentage_change])
        except KeyError:
            continue
    return np.array(data)

def build_model(input_shape):
    """
    LSTM 모델 빌드
    :param input_shape: 입력 데이터 형태
    :return: Keras 모델
    """
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)  # 예측 결과 (상승 가능성)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def predict_with_model(model, data):
    """
    모델을 사용해 예측 수행
    :param model: 학습된 모델
    :param data: 입력 데이터
    :return: 예측 결과 리스트
    """
    predictions = model.predict(data)
    return predictions.flatten()

def main():
    """
    메인 함수: 실시간 데이터 수집 및 예측 수행
    """
    exchange_names = ['binance']  # 사용할 거래소 목록
    model = build_model((3, 1))

    print("\n=== 모델 학습 중 ===")
    dummy_data = np.random.random((100, 3, 1))  # 임시 학습 데이터
    target = np.random.random((100, 1))  # 임시 타겟
    model.fit(dummy_data, target, epochs=5, verbose=0)  # 간단한 사전 학습
    print("모델 학습 완료.")

    while True:
        print("\n=== 코인 예측 및 매수 분석 시작 ===")
        for exchange_name in exchange_names:
            print(f"\n[{exchange_name.upper()} 거래소]")
            try:
                exchange = get_exchange_data(exchange_name)
                markets = exchange.load_markets()
                usdt_pairs = [symbol for symbol in markets if symbol.endswith("/USDT")]

                for symbol in usdt_pairs[:5]:  # 상위 5개 코인만 분석
                    support_msg = find_supertrend_support(exchange, symbol)
                    print(support_msg)

            except Exception as e:
                print(f"오류 발생: {e}")
        time.sleep(60)  # 1분마다 데이터 갱신

if __name__ == "__main__":
    main()
