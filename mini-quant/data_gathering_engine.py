"""
Data Gathering Engine for Mini-Quant
Multi-source data collection and management
"""

import logging
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import yfinance as yf
import time

logger = logging.getLogger(__name__)


class MarketDataProvider:
    """Market data provider using yfinance"""
    
    def get_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Get OHLCV data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            return data
        except Exception as e:
            logger.error(f"Failed to get data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_ohlcv_backup(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Backup data source"""
        # Could use Alpha Vantage or other free APIs
        return self.get_ohlcv(symbol, timeframe, start_date, end_date)


class FundamentalDataProvider:
    """Fundamental data provider"""
    
    def get_fundamentals(self, symbol: str) -> Dict:
        """Get fundamental data"""
        # Placeholder - would integrate with Financial Modeling Prep API or SEC EDGAR
        return {}


class AlternativeDataProvider:
    """Alternative data provider"""
    
    def get_sentiment(self, symbol: str) -> Dict:
        """Get sentiment data"""
        # Placeholder - would integrate with Twitter/Reddit APIs
        return {}


class NewsDataProvider:
    """News data provider"""
    
    def get_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get news articles"""
        # Placeholder - would integrate with NewsAPI.org
        return []


class SocialMediaDataProvider:
    """Social media data provider"""
    
    def get_social_sentiment(self, symbol: str) -> Dict:
        """Get social media sentiment"""
        # Placeholder - would integrate with Twitter/Reddit APIs
        return {}


class DataCache:
    """Simple data cache"""
    
    def __init__(self):
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_timestamps: Dict[str, float] = {}
        self.cache_ttl = 3600  # 1 hour
    
    def store(self, key: str, data: pd.DataFrame):
        """Store data in cache"""
        self.cache[key] = data
        self.cache_timestamps[key] = time.time()
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get data from cache"""
        if key in self.cache:
            if time.time() - self.cache_timestamps[key] < self.cache_ttl:
                return self.cache[key]
            else:
                # Expired
                del self.cache[key]
                del self.cache_timestamps[key]
        return None


class DataQualityMonitor:
    """Monitor data quality"""
    
    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data"""
        if data.empty:
            return data
        
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Remove rows with all NaN
        data = data.dropna(how='all')
        
        # Forward fill missing values (limited)
        data = data.fillna(method='ffill', limit=3)
        
        return data


class DataGatheringEngine:
    """
    Multi-source data collection and management
    
    Gathers market data from multiple free sources and manages caching.
    """
    
    def __init__(self):
        self.data_sources = {
            'market': MarketDataProvider(),
            'fundamental': FundamentalDataProvider(),
            'alternative': AlternativeDataProvider(),
            'news': NewsDataProvider(),
            'social': SocialMediaDataProvider()
        }
        self.data_cache = DataCache()
        self.data_quality_monitor = DataQualityMonitor()
        
    def gather_market_data(
        self, 
        symbols: List[str], 
        timeframe: str, 
        start_date: datetime,
        end_date: datetime, 
        region: str
    ) -> pd.DataFrame:
        """
        Gather market data from multiple sources
        
        Args:
            symbols: List of symbols to gather
            timeframe: Timeframe (e.g., '1D', '1H')
            start_date: Start date
            end_date: End date
            region: Region code (USA, EMEA, CHN, IND, etc.)
            
        Returns:
            Combined DataFrame with market data
        """
        all_data = []
        
        for symbol in symbols:
            # Check cache first
            cache_key = f"{symbol}_{region}_{timeframe}_{start_date}_{end_date}"
            cached = self.data_cache.get(cache_key)
            if cached is not None:
                all_data.append(cached)
                continue
            
            # Try primary source
            try:
                if region in ['USA', 'AMER']:
                    data = self.data_sources['market'].get_ohlcv(
                        symbol, timeframe, start_date, end_date
                    )
                elif region in ['EMEA', 'EUR']:
                    # Use symbol with exchange suffix
                    data = self.data_sources['market'].get_ohlcv(
                        f"{symbol}.L", timeframe, start_date, end_date
                    )
                elif region == 'CHN':
                    data = self.data_sources['market'].get_ohlcv(
                        f"{symbol}.SS", timeframe, start_date, end_date
                    )
                elif region == 'IND':
                    data = self.data_sources['market'].get_ohlcv(
                        f"{symbol}.BO", timeframe, start_date, end_date
                    )
                else:
                    data = self.data_sources['market'].get_ohlcv(
                        symbol, timeframe, start_date, end_date
                    )
                
                if not data.empty:
                    all_data.append(data)
                    # Cache the data
                    self.data_cache.store(cache_key, data)
                    
            except Exception as e:
                logger.warning(f"Primary source failed for {symbol}: {e}")
                # Try backup source
                try:
                    data = self.data_sources['market'].get_ohlcv_backup(
                        symbol, timeframe, start_date, end_date
                    )
                    if not data.empty:
                        all_data.append(data)
                except Exception as e2:
                    logger.error(f"Backup source also failed for {symbol}: {e2}")
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine and validate
        combined_data = pd.concat(all_data, axis=1)
        validated_data = self.data_quality_monitor.validate(combined_data)
        
        return validated_data
    
    def get_universe_symbols(self, region: str, universe: str = 'TOP3000') -> List[str]:
        """
        Get symbols for a universe in a region
        
        Args:
            region: Region code
            universe: Universe name (e.g., 'SP500', 'STOXX600')
            
        Returns:
            List of symbols
        """
        # Placeholder - would fetch from actual data sources
        # For demo, return some common symbols
        if region == 'USA' and universe == 'SP500':
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ']
        elif region == 'EMEA' and universe == 'STOXX600':
            return ['ASML', 'NOVN', 'SAP', 'LIN', 'SIE']
        elif region == 'CHN' and universe == 'CSI300':
            return ['600519', '000858', '000002']
        else:
            return ['AAPL', 'MSFT', 'GOOGL']  # Default
    
    def gather_fundamental_data(self, symbols: List[str]) -> Dict:
        """Gather fundamental data"""
        all_fundamentals = {}
        for symbol in symbols:
            fundamentals = self.data_sources['fundamental'].get_fundamentals(symbol)
            all_fundamentals[symbol] = fundamentals
        return all_fundamentals
    
    def gather_alternative_data(self, symbols: List[str]) -> Dict:
        """Gather alternative data"""
        all_alt_data = {}
        for symbol in symbols:
            sentiment = self.data_sources['alternative'].get_sentiment(symbol)
            all_alt_data[symbol] = sentiment
        return all_alt_data

