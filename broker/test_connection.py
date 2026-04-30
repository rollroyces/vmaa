#!/usr/bin/env python3
"""Test Tiger Trade API connection with paper trading account."""
import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/vmaa/broker')

from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.trade.trade_client import TradeClient
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.common.consts import SecurityType, Currency

CONFIG_PATH = '/home/node/.openclaw/workspace/vmaa/broker/tiger_openapi_config.properties'

def test_connection():
    print("=" * 60)
    print("  Tiger Trade API — Connection Test")
    print("=" * 60)

    # Load config
    config = TigerOpenClientConfig(props_path=CONFIG_PATH)
    print(f"\n[1] Config loaded")
    print(f"    Tiger ID: {config.tiger_id}")
    print(f"    Account:  {config.account}")
    print(f"    License:  {config.license}")
    print(f"    Private Key: {'✓' if config.private_key else '✗'}")

    # Test TradeClient
    trade_client = TradeClient(config)
    print(f"\n[2] TradeClient initialized")

    # Test QuoteClient
    quote_client = QuoteClient(config)
    print(f"\n[3] QuoteClient initialized")

    # Test: Get account info
    try:
        print(f"\n[4] Fetching account assets...")
        assets = trade_client.get_assets()
        print(f"    ✓ Assets fetched")
        for item in assets:
            print(f"      {item}")
    except Exception as e:
        print(f"    ✗ Account assets failed: {e}")

    # Test: Get positions
    try:
        print(f"\n[5] Fetching positions...")
        positions = trade_client.get_positions()
        print(f"    ✓ Got {len(positions)} position(s)")
        for pos in positions:
            print(f"      {pos.contract.symbol}: {pos.quantity} shares, "
                  f"Avg Cost: {pos.average_cost}, "
                  f"Mkt Price: {pos.market_price}, "
                  f"P&L: {pos.unrealized_pnl}")
    except Exception as e:
        print(f"    ✗ Positions failed: {e}")

    # Test: Get AAPL quote
    try:
        print(f"\n[6] Fetching AAPL quote...")
        contract = trade_client.get_contract('AAPL', sec_type=SecurityType.STK)
        print(f"    ✓ Contract: {contract.symbol} ({contract.name})")
        quotes = quote_client.get_stock_briefs(['AAPL'])
        for q in quotes:
            print(f"      Price: {q.latest_price}, "
                  f"Open: {q.open}, "
                  f"High: {q.high}, "
                  f"Low: {q.low}")
    except Exception as e:
        print(f"    ✗ Quote failed: {e}")

    # Test: Get orders
    try:
        print(f"\n[7] Fetching recent orders...")
        orders = trade_client.get_orders()
        print(f"    ✓ Got {len(orders)} order(s)")
        for o in orders[:5]:
            print(f"      {o.contract.symbol}: {o.action} {o.quantity} @ {o.limit_price}, "
                  f"Status: {o.status}")
    except Exception as e:
        print(f"    ✗ Orders failed: {e}")

    print(f"\n{'=' * 60}")
    print(f"  ✅ Connection test complete")
    print(f"{'=' * 60}")
    return True

if __name__ == '__main__':
    test_connection()
