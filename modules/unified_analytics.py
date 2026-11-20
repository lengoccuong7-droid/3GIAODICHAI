"""Unified analytics layer for V6.2.

One immutable-by-convention snapshot is built per symbol/data fingerprint. All UI
views consume this snapshot; no tab owns a separate calculation path.
OHLCV-derived flow/absorption labels are explicitly proxies, never participant IDs.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np
import pandas as pd
from .indicators import add_indicators, add_ohlcv_features
from .market_flow_trap import analyze_flow_and_traps
from .market_regime import assess_market_regime
from .relative_strength import relative_strength
from .data_quality import assess_data_quality
from .swing_trading_expert import analyze_swing_setup


def _fingerprint(*frames: pd.DataFrame | None) -> str:
    h = hashlib.sha256()
    for frame in frames:
        if frame is None or frame.empty:
            h.update(b'EMPTY|')
            continue
        cols = [c for c in ('date','symbol','sector','open','high','low','close','volume') if c in frame]
        x = frame[cols].copy()
        if 'date' in x:
            x['date'] = pd.to_datetime(x.date, errors='coerce').astype(str)
        h.update(pd.util.hash_pandas_object(x.astype(str), index=False).values.tobytes())
    return h.hexdigest()[:24]


def enrich_once(raw: pd.DataFrame) -> pd.DataFrame:
    """Canonical indicator/feature calculation. Do not recalculate in tabs."""
    return add_ohlcv_features(add_indicators(raw.sort_values('date').copy()))


def analyze_accumulation_distribution(x: pd.DataFrame) -> dict:
    """Rule-based multi-session structure assessment from OHLCV, not proof of intent."""
    if x is None or len(x) < 40:
        return {'status':'INSUFFICIENT_DATA','label':'CHƯA ĐỦ DỮ LIỆU','score':None,
                'confidence':'THẤP','evidence':[],'cautions':['Cần tối thiểu 40 phiên; khuyến nghị 60–120 phiên.']}
    z = x.tail(40).copy()
    last = z.iloc[-1]
    clv = z.CLV if 'CLV' in z else (((z.close-z.low)-(z.high-z.close))/(z.high-z.low).replace(0,np.nan)).fillna(0)
    rvol = float(last.REL_VOLUME20) if pd.notna(last.get('REL_VOLUME20')) else None
    net_clv = float(clv.tail(20).mean())
    up_vol = float(z.loc[z.close.diff()>0,'volume'].tail(20).sum())
    down_vol = float(z.loc[z.close.diff()<0,'volume'].tail(20).sum())
    vol_balance = (up_vol-down_vol)/(up_vol+down_vol) if up_vol+down_vol>0 else 0.0
    range20 = float((z.high.tail(20).max()-z.low.tail(20).min())/max(z.close.iloc[-20],1e-9)*100)
    range_prev = float((z.high.iloc[:20].max()-z.low.iloc[:20].min())/max(z.close.iloc[0],1e-9)*100)
    contraction = range20 < range_prev * .8 if range_prev>0 else False
    down_days = z.close.diff()<0
    down_volume_share = float(z.loc[down_days,'volume'].tail(20).sum()/max(z.volume.tail(20).sum(),1))
    score = 0; evidence=[]
    if net_clv >= .12: score += 1; evidence.append('CLV bình quân 20 phiên nghiêng về nửa trên biên độ giá.')
    elif net_clv <= -.12: score -= 1; evidence.append('CLV bình quân 20 phiên nghiêng về nửa dưới biên độ giá.')
    if vol_balance >= .10: score += 1; evidence.append('Khối lượng phiên tăng trội hơn phiên giảm trong cửa sổ quan sát.')
    elif vol_balance <= -.10: score -= 1; evidence.append('Khối lượng phiên giảm trội hơn phiên tăng trong cửa sổ quan sát.')
    if contraction: score += 1; evidence.append('Biên độ 20 phiên gần đây co hẹp so với 20 phiên trước.')
    if rvol is not None and rvol >= 1.4 and abs(float(last.high-last.low)/max(float(last.close),1e-9)) <= .025:
        score += 1 if float(last.CLV)>=.15 else -1 if float(last.CLV)<=-.15 else 0
        evidence.append('Phiên khối lượng cao nhưng biên độ hẹp: cần đọc cùng vị trí đóng cửa, không tự kết luận hấp thụ.')
    if down_volume_share >= .62 and net_clv < 0: score -= 1
    if score >= 3:
        label, confidence = 'NỀN TÍCH LŨY / HẤP THỤ TIỀM NĂNG','TRUNG BÌNH'
    elif score <= -2:
        label, confidence = 'ÁP LỰC PHÂN PHỐI TIỀM NĂNG','TRUNG BÌNH'
    else:
        label, confidence = 'CHƯA ĐỦ BẰNG CHỨNG TÍCH LŨY / PHÂN PHỐI','THẤP'
    return {'status':'OK','label':label,'score':int(score),'confidence':confidence,
            'metrics':{'clv_mean_20':net_clv,'up_down_volume_balance':vol_balance,
                       'range20_pct':range20,'range_previous20_pct':range_prev,
                       'down_day_volume_share_20':down_volume_share,'relative_volume20':rvol},
            'evidence':evidence,
            'cautions':['Đây là phân loại hành vi giá/khối lượng, không chứng minh tổ chức đang gom hoặc xả.',
                        'Cần xác nhận bằng cấu trúc giá, bối cảnh thị trường và các phiên tiếp theo.']}


def evaluate_unified_decision(enriched: pd.DataFrame, market_regime: dict | None = None,
                              rs_market: dict | None = None, rs_sector: dict | None = None,
                              data_quality: dict | None = None, capital_vnd: float = 100_000_000,
                              risk_pct: float = 1.0, max_alloc_pct: float = 20.0,
                              lot_size: int = 100, position: dict | None = None,
                              fee_pct: float = .15, slippage_pct: float = .10,
                              tax_pct: float = .10) -> tuple[dict, dict, dict]:
    """The one decision path used by live analysis and historical replay."""
    quality = data_quality or assess_data_quality(enriched)
    flow = analyze_flow_and_traps(enriched)
    structure = analyze_accumulation_distribution(enriched)
    flow['structure_analysis'] = structure
    report = analyze_swing_setup(enriched, market_regime=market_regime, rs_market=rs_market,
        rs_sector=rs_sector, capital_vnd=capital_vnd, risk_pct=risk_pct,
        max_allocation_pct=max_alloc_pct, lot_size=lot_size, position=position,
        data_quality=quality, flow_trap=flow, fee_pct=fee_pct,
        slippage_pct=slippage_pct, tax_pct=tax_pct)
    if structure['score'] is not None:
        report['structure_analysis'] = structure
        if structure['score'] <= -2:
            msg = 'Cấu trúc OHLCV nghiêng về phân phối; cần xác nhận trước khi xem xét vị thế mới.'
            if msg not in report['blockers']: report['blockers'].append(msg)
            if report['status'] == 'TRADE_PLAN_READY':
                report['status'] = 'WAIT_CONFIRMATION'
                report['decision'] = report['conclusion'] = 'CHỜ XÁC NHẬN – CẤU TRÚC CUNG CẦN ĐƯỢC KIỂM TRA'
        elif structure['score'] >= 3:
            msg = 'Cấu trúc giá/khối lượng có dấu hiệu tích lũy hoặc hấp thụ tiềm năng; chưa phải bằng chứng tổ chức gom hàng.'
            if msg not in report['positives']: report['positives'].append(msg)
    return report, flow, structure


def build_unified_snapshot(symbol: str, raw: pd.DataFrame, universe: pd.DataFrame | None = None,
                           benchmark: pd.DataFrame | None = None, sector: pd.DataFrame | None = None,
                           capital_vnd: float = 100_000_000, risk_pct: float = 1.0,
                           max_alloc_pct: float = 20.0, lot_size: int = 100,
                           position: dict | None = None, fee_pct: float = .15,
                           slippage_pct: float = .10, tax_pct: float = .10) -> dict:
    symbol = str(symbol).strip().upper()
    if raw is None or raw.empty or symbol not in set(raw.symbol.astype(str).str.upper()):
        raise ValueError('Mã phân tích không khớp dữ liệu OHLCV.')
    enriched = enrich_once(raw)
    quality = assess_data_quality(raw)
    asof = pd.to_datetime(raw.date, errors='coerce').max()
    benchmark_asof = None
    if benchmark is not None and not benchmark.empty:
        benchmark_asof = benchmark[pd.to_datetime(benchmark.date, errors='coerce') <= asof].copy()
    sector_asof = None
    if sector is not None and not sector.empty:
        sector_asof = sector[pd.to_datetime(sector.date, errors='coerce') <= asof].copy()
    market_frame = universe.copy() if universe is not None else raw.copy()
    if 'date' in market_frame:
        market_frame = market_frame[pd.to_datetime(market_frame.date, errors='coerce') <= asof]
    regime = assess_market_regime(market_frame if benchmark_asof is None else benchmark_asof)
    rs_market = relative_strength(raw, benchmark_asof, 20) if benchmark_asof is not None and not benchmark_asof.empty else {}
    rs_sector = relative_strength(raw, sector_asof, 20) if sector_asof is not None and not sector_asof.empty else {}
    report, flow, structure = evaluate_unified_decision(
        enriched, market_regime=regime, rs_market=rs_market, rs_sector=rs_sector,
        data_quality=quality, capital_vnd=capital_vnd, risk_pct=risk_pct,
        max_alloc_pct=max_alloc_pct, lot_size=lot_size, position=position,
        fee_pct=fee_pct, slippage_pct=slippage_pct, tax_pct=tax_pct)
    fp = _fingerprint(raw, benchmark_asof, sector_asof)
    config = {'capital_vnd':float(capital_vnd),'risk_pct':float(risk_pct),
              'max_alloc_pct':float(max_alloc_pct),'lot_size':int(lot_size),
              'position':position or {},'fee_pct':float(fee_pct),
              'slippage_pct':float(slippage_pct),'tax_pct':float(tax_pct)}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]
    analysis_id = hashlib.sha256(f'{symbol}|{fp}|{config_hash}|V6.2.0'.encode()).hexdigest()[:24]
    return {'schema_version':'6.2.0','engine_version':'UNIFIED-1.0','analysis_id':analysis_id,
            'data_fingerprint':fp,'config_fingerprint':config_hash,'analysis_config':config,
            'symbol':symbol,'as_of':str(pd.to_datetime(raw.date).max()),
            'enriched':enriched,'data_quality':quality,'market_regime':regime,
            'rs_market':rs_market,'rs_sector':rs_sector,'flow_trap':flow,
            'structure_analysis':structure,'swing_report':report,
            'calculation_contract':{'indicators':'modules/indicators.py::add_indicators',
              'ohlcv_features':'modules/unified_analytics.py::enrich_once',
              'snapshot_id':analysis_id,'shared_by_tabs':True}}
