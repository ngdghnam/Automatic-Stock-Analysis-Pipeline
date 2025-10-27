"""
HỆ THỐNG HỖ TRỢ RA QUYẾT ĐỊNH ĐẦU TƯ
Investment Decision Support System (IDSS)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')

# Technical Analysis
import ta
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

# Machine Learning
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error

# ====================================================================
# PHẦN 1: THU THẬP DỮ LIỆU
# ====================================================================

class DataCollector:
    """Class để thu thập và tổng hợp dữ liệu từ nhiều nguồn"""
    
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        
    def get_price_data(self):
        """Lấy dữ liệu giá lịch sử"""
        from vnstock3 import Quote
        quote = Quote(symbol=self.symbol, source='VCI')
        df = quote.history(start=self.start_date, end=self.end_date)
        df['symbol'] = self.symbol
        return df
    
    def get_company_fundamentals(self):
        """Lấy dữ liệu cơ bản công ty"""
        from vnstock3 import Company
        company = Company(symbol=self.symbol, source='VCI')
        
        # Overview
        overview = company.overview()
        overview = overview.drop_duplicates(subset=['symbol'])
        
        # Financial ratios
        ratios = company.ratio_summary()
        if not ratios.empty:
            ratios['year_report'] = ratios['year_report'].astype(int)
        
        # Trading stats
        trading = company.trading_stats()
        trading = trading.drop_duplicates(subset=['symbol'])
        
        return overview, ratios, trading
    
    def get_market_data(self):
        """Lấy dữ liệu thị trường (VN-Index)"""
        from vnstock3 import Quote
        try:
            vnindex = Quote(symbol='VNINDEX', source='VCI')
            market_df = vnindex.history(start=self.start_date, end=self.end_date)
            market_df = market_df.rename(columns={
                'close': 'market_close',
                'volume': 'market_volume'
            })
            market_df = market_df[['time', 'market_close', 'market_volume']]
            return market_df
        except:
            print("⚠️ Không thể lấy dữ liệu VN-Index")
            return pd.DataFrame()
    
    def merge_all_data(self):
        """Kết hợp tất cả dữ liệu"""
        print(f"📊 Thu thập dữ liệu cho {self.symbol}...")
        
        # 1. Dữ liệu giá
        price_df = self.get_price_data()
        price_df['year_report'] = pd.to_datetime(price_df['time']).dt.year
        print(f"✓ Giá: {len(price_df)} ngày")
        
        # 2. Dữ liệu thị trường
        market_df = self.get_market_data()
        if not market_df.empty:
            price_df = pd.merge(price_df, market_df, on='time', how='left')
            print(f"✓ Thị trường: {len(market_df)} ngày")
        
        # 3. Dữ liệu công ty
        overview, ratios, trading = self.get_company_fundamentals()
        
        # Merge overview
        if not overview.empty:
            cols_to_drop = [c for c in overview.columns 
                          if c in price_df.columns and c != 'symbol']
            overview = overview.drop(columns=cols_to_drop)
            price_df = pd.merge(price_df, overview, on='symbol', how='left')
            print(f"✓ Tổng quan: {len(overview.columns)} features")
        
        # Merge ratios
        if not ratios.empty:
            cols_to_drop = [c for c in ratios.columns 
                          if c in price_df.columns and c not in ['symbol', 'year_report']]
            ratios = ratios.drop(columns=cols_to_drop)
            price_df = pd.merge(price_df, ratios, 
                              on=['symbol', 'year_report'], how='left')
            print(f"✓ Tài chính: {len(ratios.columns)} features")
        
        # Merge trading
        if not trading.empty:
            cols_to_drop = [c for c in trading.columns 
                          if c in price_df.columns and c != 'symbol']
            trading = trading.drop(columns=cols_to_drop)
            price_df = pd.merge(price_df, trading, on='symbol', how='left')
            print(f"✓ Giao dịch: {len(trading.columns)} features")
        
        self.data = price_df
        print(f"\n✅ Tổng cộng: {price_df.shape[0]} rows × {price_df.shape[1]} columns")
        return self.data


# ====================================================================
# PHẦN 2: FEATURE ENGINEERING
# ====================================================================

class FeatureEngineer:
    """Class để tạo các features từ dữ liệu thô"""
    
    def __init__(self, df):
        self.df = df.copy()
        
    def add_technical_indicators(self):
        """Thêm các chỉ báo kỹ thuật"""
        print("\n🔧 Tạo Technical Indicators...")
        df = self.df.copy()
        
        # 1. Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_change'] = df['close'] - df['open']
        df['high_low_range'] = df['high'] - df['low']
        
        # 2. Moving Averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # 3. RSI
        rsi = RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi.rsi()
        
        # 4. MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # 5. Bollinger Bands
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        
        # 6. Stochastic
        stoch = StochasticOscillator(high=df['high'], low=df['low'], 
                                      close=df['close'], window=14)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # 7. ATR (Average True Range)
        atr = AverageTrueRange(high=df['high'], low=df['low'], 
                                close=df['close'], window=14)
        df['atr'] = atr.average_true_range()
        
        # 8. Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        print(f"✓ Đã tạo {df.shape[1] - self.df.shape[1]} technical indicators")
        self.df = df
        return self
    
    def add_lag_features(self, periods=[1, 2, 3, 5, 10]):
        """Thêm features từ quá khứ"""
        print("\n🔧 Tạo Lag Features...")
        df = self.df.copy()
        
        for period in periods:
            df[f'close_lag_{period}'] = df['close'].shift(period)
            df[f'volume_lag_{period}'] = df['volume'].shift(period)
            df[f'returns_lag_{period}'] = df['returns'].shift(period)
        
        print(f"✓ Đã tạo {len(periods) * 3} lag features")
        self.df = df
        return self
    
    def add_rolling_features(self, windows=[5, 10, 20]):
        """Thêm rolling statistics"""
        print("\n🔧 Tạo Rolling Features...")
        df = self.df.copy()
        
        for window in windows:
            # Rolling statistics for price
            df[f'close_rolling_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'close_rolling_std_{window}'] = df['close'].rolling(window).std()
            df[f'close_rolling_min_{window}'] = df['close'].rolling(window).min()
            df[f'close_rolling_max_{window}'] = df['close'].rolling(window).max()
            
            # Rolling statistics for volume
            df[f'volume_rolling_mean_{window}'] = df['volume'].rolling(window).mean()
            
            # Volatility
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
        
        print(f"✓ Đã tạo {len(windows) * 6} rolling features")
        self.df = df
        return self
    
    def add_time_features(self):
        """Thêm time-based features"""
        print("\n🔧 Tạo Time Features...")
        df = self.df.copy()
        
        df['time'] = pd.to_datetime(df['time'])
        df['day_of_week'] = df['time'].dt.dayofweek
        df['day_of_month'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        df['quarter'] = df['time'].dt.quarter
        df['is_month_start'] = df['time'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['time'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['time'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['time'].dt.is_quarter_end.astype(int)
        
        print(f"✓ Đã tạo 8 time features")
        self.df = df
        return self
    
    def create_target_variable(self, horizon=1, method='classification', threshold=0.01):
        """
        Tạo biến target cho dự đoán
        
        Parameters:
        -----------
        horizon: int - Dự đoán sau bao nhiêu ngày
        method: str - 'classification' hoặc 'regression'
        threshold: float - Ngưỡng để phân loại tăng/giảm (cho classification)
        """
        print(f"\n🎯 Tạo Target Variable (method={method}, horizon={horizon})...")
        df = self.df.copy()
        
        if method == 'classification':
            # Binary: Tăng (1) hoặc Giảm (0)
            df['future_returns'] = df['close'].shift(-horizon) / df['close'] - 1
            df['target'] = (df['future_returns'] > threshold).astype(int)
            print(f"✓ Target: 0=Giảm, 1=Tăng (threshold={threshold*100}%)")
            print(f"  - Tăng: {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
            print(f"  - Giảm: {(1-df['target']).sum()} ({(1-df['target'].mean())*100:.1f}%)")
            
        elif method == 'regression':
            # Dự đoán giá tương lai
            df['target'] = df['close'].shift(-horizon)
            print(f"✓ Target: Giá sau {horizon} ngày")
        
        self.df = df
        return self
    
    def get_processed_data(self):
        """Trả về dữ liệu đã xử lý"""
        return self.df


# ====================================================================
# PHẦN 3: PHÂN TÍCH DỮ LIỆU
# ====================================================================

class DataAnalyzer:
    """Class để phân tích và visualize dữ liệu"""
    
    def __init__(self, df):
        self.df = df
        
    def data_summary(self):
        """Tổng quan về dữ liệu"""
        print("\n" + "="*70)
        print("📊 DATA SUMMARY")
        print("="*70)
        
        print(f"\n📏 Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"📅 Time range: {self.df['time'].min()} to {self.df['time'].max()}")
        print(f"💰 Price range: {self.df['close'].min():.2f} - {self.df['close'].max():.2f}")
        
        # Missing values
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing': missing,
            'Percent': missing_pct
        })
        missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
        
        if len(missing_df) > 0:
            print(f"\n⚠️ Missing values: {len(missing_df)} columns")
            print(missing_df.head(10))
        else:
            print("\n✅ No missing values")
        
        return missing_df
    
    def plot_price_and_indicators(self, figsize=(15, 12)):
        """Vẽ biểu đồ giá và các chỉ báo"""
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # 1. Price and Moving Averages
        axes[0].plot(self.df['time'], self.df['close'], label='Close', linewidth=2)
        if 'sma_20' in self.df.columns:
            axes[0].plot(self.df['time'], self.df['sma_20'], label='SMA 20', alpha=0.7)
        if 'sma_50' in self.df.columns:
            axes[0].plot(self.df['time'], self.df['sma_50'], label='SMA 50', alpha=0.7)
        axes[0].set_title('Price Chart with Moving Averages', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Volume
        axes[1].bar(self.df['time'], self.df['volume'], alpha=0.5)
        axes[1].set_title('Trading Volume', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # 3. RSI
        if 'rsi' in self.df.columns:
            axes[2].plot(self.df['time'], self.df['rsi'])
            axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.5)
            axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.5)
            axes[2].set_title('RSI (Relative Strength Index)', fontsize=12, fontweight='bold')
            axes[2].set_ylim(0, 100)
            axes[2].grid(True, alpha=0.3)
        
        # 4. MACD
        if 'macd' in self.df.columns:
            axes[3].plot(self.df['time'], self.df['macd'], label='MACD')
            axes[3].plot(self.df['time'], self.df['macd_signal'], label='Signal')
            axes[3].bar(self.df['time'], self.df['macd_diff'], label='Histogram', alpha=0.3)
            axes[3].set_title('MACD', fontsize=12, fontweight='bold')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def correlation_analysis(self, top_n=20):
        """Phân tích correlation"""
        # Chọn các features số
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if 'target' in numeric_cols:
            # Correlation với target
            corr_with_target = self.df[numeric_cols].corr()['target'].abs().sort_values(ascending=False)
            
            print("\n" + "="*70)
            print(f"📊 TOP {top_n} FEATURES TƯƠNG QUAN VỚI TARGET")
            print("="*70)
            print(corr_with_target.head(top_n))
            
            # Visualize
            plt.figure(figsize=(10, 8))
            corr_with_target.head(top_n).plot(kind='barh')
            plt.title(f'Top {top_n} Features Correlated with Target', fontsize=12, fontweight='bold')
            plt.xlabel('Absolute Correlation')
            plt.tight_layout()
            plt.show()
            
            return corr_with_target
        else:
            print("⚠️ Target variable chưa được tạo")
            return None


# ====================================================================
# PHẦN 4: MÔ HÌNH DỰ ĐOÁN
# ====================================================================

class PredictionModel:
    """Class để training và đánh giá mô hình"""
    
    def __init__(self, df):
        self.df = df
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_data(self, test_size=0.2, features_to_exclude=None):
        """
        Chuẩn bị dữ liệu cho training
        """
        print("\n" + "="*70)
        print("🔧 CHUẨN BỊ DỮ LIỆU CHO TRAINING")
        print("="*70)
        
        df = self.df.copy()
        
        # Loại bỏ các dòng có target = NaN
        df = df.dropna(subset=['target'])
        print(f"✓ Sau khi loại NaN: {len(df)} samples")
        
        # Xác định features và target
        if features_to_exclude is None:
            features_to_exclude = ['time', 'symbol', 'target', 'future_returns']
        
        # Chỉ lấy các cột số
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols 
                       if col not in features_to_exclude]
        
        X = df[feature_cols].copy()
        y = df['target'].copy()
        
        # Xử lý missing values trong features
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"✓ Features: {X.shape[1]} columns")
        print(f"✓ Target: {y.shape[0]} samples")
        print(f"✓ Target distribution: {y.value_counts().to_dict()}")
        
        # Split data (time-series aware)
        split_idx = int(len(X) * (1 - test_size))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.feature_names = feature_cols
        
        print(f"\n✓ Train set: {X_train.shape[0]} samples")
        print(f"✓ Test set: {X_test.shape[0]} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train, model_type='random_forest'):
        """Training mô hình"""
        print(f"\n🤖 TRAINING MODEL: {model_type.upper()}")
        print("="*70)
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        
        self.model.fit(X_train, y_train)
        print("✅ Training completed!")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Đánh giá mô hình"""
        print("\n" + "="*70)
        print("📊 MÔ HÌNH ĐÁNH GIÁ")
        print("="*70)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Classification report
        print("\n" + classification_report(y_test, y_pred, 
                                          target_names=['Giảm', 'Tăng']))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_imp = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n📊 TOP 20 IMPORTANT FEATURES:")
            print(feature_imp.head(20))
            
            # Plot
            plt.figure(figsize=(10, 8))
            feature_imp.head(20).plot(x='feature', y='importance', kind='barh')
            plt.title('Top 20 Feature Importances', fontsize=12, fontweight='bold')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
        
        return y_pred


# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main():
    """Hàm chính để chạy toàn bộ pipeline"""
    
    print("="*70)
    print(" HỆ THỐNG HỖ TRỢ RA QUYẾT ĐỊNH ĐẦU TƯ".center(70))
    print(" Investment Decision Support System (IDSS)".center(70))
    print("="*70)
    
    # Config
    SYMBOL = "ACB"
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    START_DATE = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 năm
    
    print(f"\n📌 Symbol: {SYMBOL}")
    print(f"📅 Period: {START_DATE} to {END_DATE}")
    
    # Step 1: Thu thập dữ liệu
    collector = DataCollector(SYMBOL, START_DATE, END_DATE)
    data = collector.merge_all_data()
    
    # Step 2: Feature Engineering
    engineer = FeatureEngin