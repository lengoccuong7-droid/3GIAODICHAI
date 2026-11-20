import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from config import APP_NAME, DEFAULT_CAPITAL_VND, APP_VERSION, PRICE_UNIT_OPTIONS, DEFAULT_PRICE_UNIT, DEFAULT_TAX_PCT
from modules.data_loader import load_market_file
from modules.data_store import upsert_market_data, load_market_history, market_history_summary
from modules.data_ingestion import prepare_market_upload
from modules.support_resistance import find_support_resistance
from modules.data_quality import assess_data_quality
from modules.backtest import simple_swing_backtest
from modules.risk_manager import calculate_position_size
from modules.relative_strength import relative_strength
from modules.market_analysis import analyze_symbol, analyze_universe
from modules.stock_scanner import scan_universe
from modules.market_regime import assess_market_regime
from modules.multi_timeframe import timeframe_snapshot
from modules.fundamental_analysis import valuation_metrics, quality_flags
from modules.assistant_reasoning import build_reasoning_report
from modules.knowledge_hub import index_document, search_documents, list_documents
from modules.master_counsel import build_master_counsel, answer_counsel_question, save_decision_record, list_decision_records
from modules.swing_trading_expert import analyze_swing_setup, scan_swing_universe
from modules.unified_analytics import build_unified_snapshot
from modules.market_flow_trap import analyze_flow_and_traps
from modules.plan_backtest import backtest_plan
from modules.beginner_summary import build_beginner_summary
from modules.market_diagnostics import classify_stock_phase, market_status_label

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1500px}
[data-testid="stMetricValue"]{font-size:1.35rem}
.stTabs [data-baseweb="tab-list"]{gap:.25rem;flex-wrap:nowrap;overflow-x:auto;position:sticky;top:0;z-index:9999;background:var(--background-color);padding:.45rem .15rem;border-bottom:1px solid rgba(128,128,128,.25);scrollbar-width:thin;isolation:isolate}
.stTabs [data-baseweb="tab"]{white-space:nowrap;padding:.48rem .68rem;min-height:2.45rem}
.stTabs [data-baseweb="tab-panel"]{padding-top:1rem;scroll-margin-top:4rem}
[data-testid="stAlert"]{padding:.55rem .75rem;border-radius:10px}
[data-testid="stJson"]{max-width:100%;overflow-x:auto}
@media(max-width:700px){.stTabs [data-baseweb="tab-list"]{top:0;position:sticky}.block-container{padding-left:.65rem;padding-right:.65rem;padding-top:.55rem}.stTabs [data-baseweb="tab-list"]{top:0}.stTabs [data-baseweb="tab"]{font-size:.76rem;padding:.4rem .55rem}.stTabs [data-baseweb="tab-panel"]{padding-left:0;padding-right:0}.stMetric{padding:.35rem}.signal-card{padding:.7rem}}

.signal-card{padding:1rem;border-radius:12px;background:rgba(128,128,128,.10);border:1px solid rgba(128,128,128,.25)}
.desk-kicker{font-size:.72rem;letter-spacing:.16em;font-weight:700;color:#65b8a5;margin-bottom:-.45rem}
[data-testid="stSidebar"]{border-right:1px solid rgba(128,128,128,.18)}
[data-testid="stMetric"]{background:rgba(128,128,128,.07);padding:.75rem;border-radius:12px;border:1px solid rgba(128,128,128,.14)}
.stButton>button[kind="primary"]{font-weight:700;letter-spacing:.02em}
@media(max-width:700px){[data-testid="stMetric"]{padding:.55rem}.desk-kicker{font-size:.65rem}}
</style>
""", unsafe_allow_html=True)
st.markdown("<div class='desk-kicker'>TRADING DESK · SWING 3–10 PHIÊN</div>", unsafe_allow_html=True)
st.title(APP_NAME)
st.caption(f"V{APP_VERSION} · Phân tích dữ liệu thủ công · Mô phỏng, không đặt lệnh tự động")
st.caption("Nạp dữ liệu bằng CSV/Excel hoặc dán bảng tại thanh bên. Kho lịch sử được lưu cục bộ để cập nhật nối tiếp.")

# -------------------- Helpers --------------------
def fmt(v, digits=2):
    try:
        if v is None or pd.isna(v) or not np.isfinite(float(v)): return "Chưa đủ dữ liệu"
        return f"{float(v):,.{digits}f}"
    except Exception: return "Chưa đủ dữ liệu"

def read_upload(up, hint=None):
    if up is None: return None
    return load_market_file(up, symbol_hint=hint)

def annualized_gk(x):
    if len(x)<20:return None
    term=.5*np.log(x.high/x.low)**2-(2*np.log(2)-1)*np.log(x.close/x.open)**2
    v=term.replace([np.inf,-np.inf],np.nan).dropna()
    return float(np.sqrt(max(v.mean(),0)*252)*100) if len(v)>=10 else None

def annualized_parkinson(x):
    if len(x)<20:return None
    v=(np.log(x.high/x.low)**2).replace([np.inf,-np.inf],np.nan).dropna()
    return float(np.sqrt(v.mean()/(4*np.log(2))*252)*100) if len(v)>=10 else None

def hurst_rs(series):
    s=pd.Series(series).dropna().astype(float).values
    if len(s)<100:return None
    lags=[8,16,32,64]
    vals=[]
    for lag in lags:
        chunks=[]
        for i in range(0,len(s)-lag+1,lag):
            z=s[i:i+lag]; z=z-z.mean(); sd=z.std()
            if sd>0: chunks.append((z.cumsum().max()-z.cumsum().min())/sd)
        if chunks: vals.append((lag,float(np.mean(chunks))))
    if len(vals)<3:return None
    return float(np.polyfit(np.log([a for a,b in vals]),np.log([b for a,b in vals]),1)[0])

def return_entropy(series, bins=10):
    s=pd.Series(series).dropna().astype(float)
    if len(s)<50:return None
    counts=np.histogram(s,bins=bins)[0].astype(float)
    p=counts[counts>0]/counts.sum()
    return float(-(p*np.log(p)).sum()/np.log(bins))

def audit_flags(x):
    flags=[]
    if x.empty:return flags
    last=x.iloc[-1]
    if len(x)>1:
        prev=x.iloc[-2]
        if last.close>0 and prev.close>0 and abs(last.close/prev.close-1)>.25:
            flags.append("Biến động giá ngày gần nhất vượt 25%: kiểm tra điều chỉnh giá, chia tách hoặc dữ liệu lỗi.")
    if (x.volume==0).any(): flags.append(f"Có {(x.volume==0).sum()} phiên khối lượng bằng 0.")
    if x.date.duplicated().any(): flags.append("Có ngày giao dịch trùng lặp.")
    if x.date.is_monotonic_increasing is False: flags.append("Ngày chưa sắp xếp tăng dần.")
    return flags

# -------------------- Compact manual data desk --------------------
DATA_DB_PATH = Path(__file__).parent / "data" / "market_history.sqlite3"
SECTOR_DATA_PATH = Path(__file__).parent / "data" / "sector_index.csv"

with st.sidebar:
    st.header("NẠP DỮ LIỆU")
    st.caption("Nhập thủ công · Không kết nối nguồn tự động · Một kho dữ liệu chuẩn")
    input_mode = st.radio(
        "Cách nhập",
        ["Tải nhiều CSV / Excel", "Dán bảng dữ liệu"],
        key="data_input_mode",
        horizontal=False,
    )
    market_uploads = []
    pasted_text = ""
    if input_mode == "Tải nhiều CSV / Excel":
        st.markdown("**CẬP NHẬT DỮ LIỆU OHLCV**")
        st.caption("Chọn file dữ liệu cổ phiếu và VNINDEX; có thể tải nhiều file cùng lúc.")
        market_uploads = st.file_uploader(
            "Dữ liệu OHLCV",
            type=["csv", "xlsx", "xls", "txt"],
            accept_multiple_files=True,
            key="market_history_uploads",
            help=(
                "Mỗi file cần có ngày, mã, mở, cao, thấp, đóng cửa, khối lượng. "
                "File nhiều mã cần có cột symbol. Có thể tải nhiều file cùng lúc."
            ),
        )
    else:
        pasted_text = st.text_area(
            "Dán bảng OHLCV", height=150, key="market_history_paste",
            placeholder="date,symbol,open,high,low,close,volume\n2026-11-18,DXG,..."
        )
    symbol_hint = st.text_input(
        "Mã cổ phiếu (chỉ nhập nếu tệp không có cột mã)", "", key="symbol_hint",
        placeholder="Ví dụ: DXG"
    )
    with st.expander("Tùy chọn nhập dữ liệu", expanded=False):
        price_unit = st.selectbox(
            "Đơn vị giá cổ phiếu trong tệp", list(PRICE_UNIT_OPTIONS),
            index=list(PRICE_UNIT_OPTIONS).index(DEFAULT_PRICE_UNIT), key="input_price_unit",
            help="Chỉ áp dụng cho cổ phiếu. VNINDEX được giữ nguyên điểm số."
        )
        st.caption("Chỉ cần đổi nếu đơn vị giá trong tệp khác lựa chọn mặc định. VNINDEX không bị quy đổi.")

    incoming = None
    parse_error = None
    try:
        frames = []
        if market_uploads:
            for up in market_uploads:
                # Symbol hint is only safe for a single file; multi-file batches must identify each ticker.
                hint = symbol_hint.strip() or None if len(market_uploads) == 1 else None
                frames.append(load_market_file(up, sheet_name=0, symbol_hint=hint))
        elif pasted_text.strip():
            pasted_file = SimpleNamespace(name="pasted_market_data.csv", getvalue=lambda: pasted_text.encode("utf-8-sig"))
            frames.append(load_market_file(pasted_file, sheet_name=0, symbol_hint=symbol_hint.strip() or None))
        if frames:
            incoming = pd.concat(frames, ignore_index=True)
            if incoming.duplicated(["symbol", "date"]).any():
                raise ValueError("Các file có mã/ngày trùng nhau. Hãy loại bản ghi trùng trước khi cập nhật.")
    except Exception as exc:
        parse_error = str(exc)
    if parse_error:
        st.error(f"Không đọc được dữ liệu: {parse_error}")

    if incoming is not None and not incoming.empty:
        st.caption(f"Đã đọc {len(incoming):,} dòng · {incoming.symbol.nunique()} mã · {incoming.date.min():%d/%m/%Y} – {incoming.date.max():%d/%m/%Y}")
        if st.button("CẬP NHẬT DỮ LIỆU VÀO KHO", type="primary", use_container_width=True, key="commit_market_data"):
            try:
                staged, ingest_meta = prepare_market_upload(
                    incoming, price_multiplier=PRICE_UNIT_OPTIONS[price_unit]
                )
                result = upsert_market_data(staged, DATA_DB_PATH)
                st.success(
                    f"Đã cập nhật kho: thêm {result['inserted']:,} phiên · sửa {result['updated']:,} phiên · "
                    f"không đổi {result['unchanged']:,} phiên."
                )
            except Exception as exc:
                st.error(f"Không thể cập nhật kho dữ liệu: {exc}")

    st.divider()
    st.subheader("DỮ LIỆU NGÀNH (TÙY CHỌN)")
    sector_upload = st.file_uploader("Tải CSV chỉ số ngành", type=["csv"], key="sector_index_upload",
        help="Cột bắt buộc: date, sector, close. Có thể thêm volume. Đây là chuỗi chỉ số ngành do bạn cung cấp.")
    if sector_upload is not None:
        try:
            sf = pd.read_csv(sector_upload)
            sf.columns = [str(c).strip().lower() for c in sf.columns]
            aliases = {"ngay":"date", "ngày":"date", "nganh":"sector", "ngành":"sector", "dong cua":"close", "đóng cửa":"close", "gia dong cua":"close", "khối lượng":"volume", "khoi luong":"volume"}
            sf = sf.rename(columns={c:aliases.get(c,c) for c in sf.columns})
            if not {"date","sector","close"}.issubset(sf.columns):
                raise ValueError("File ngành cần có ít nhất 3 cột: date, sector, close.")
            sf["date"] = pd.to_datetime(sf["date"], dayfirst=True, errors="coerce")
            sf["close"] = pd.to_numeric(sf["close"], errors="coerce")
            sf["sector"] = sf["sector"].astype(str).str.strip().str.upper()
            sf = sf.dropna(subset=["date","close"])
            if sf.empty or (sf.close <= 0).any(): raise ValueError("Dữ liệu ngành rỗng hoặc có giá trị close không hợp lệ.")
            sf = sf.sort_values(["sector","date"])
            st.session_state["sector_index_frame"] = sf
            SECTOR_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            sf.to_csv(SECTOR_DATA_PATH, index=False, encoding="utf-8-sig")
            st.success(f"Đã lưu cục bộ dữ liệu {sf.sector.nunique()} ngành · {len(sf)} dòng. Chọn ngành tương ứng ở dưới.")
        except Exception as exc:
            st.error(f"Không đọc được file ngành: {exc}")
    sf_saved = st.session_state.get("sector_index_frame")
    if sf_saved is None and SECTOR_DATA_PATH.exists():
        try:
            sf_saved = pd.read_csv(SECTOR_DATA_PATH)
            sf_saved["date"] = pd.to_datetime(sf_saved["date"], errors="coerce")
            st.session_state["sector_index_frame"] = sf_saved
        except Exception:
            sf_saved = None
    selected_sector_name = None
    if isinstance(sf_saved, pd.DataFrame) and not sf_saved.empty:
        sector_names = sorted(sf_saved.sector.unique().tolist())
        selected_sector_name = st.selectbox("Ngành của mã đang phân tích", sector_names, key="selected_sector_name")

    store_summary = market_history_summary(DATA_DB_PATH)
    if store_summary["rows"]:
        st.caption(
            f"Kho cục bộ: {store_summary['symbols']} mã · {store_summary['rows']:,} dòng · "
            f"mới nhất {pd.to_datetime(store_summary['latest_date']).strftime('%d/%m/%Y')}"
        )
    else:
        st.info("Kho dữ liệu đang trống. Hãy nạp tệp có OHLCV để bắt đầu.")

    st.divider()
    st.subheader("VỊ THẾ MÔ PHỎNG")
    position_mode = st.radio(
        "Vị thế mô phỏng", ["CHƯA CÓ HÀNG", "ĐANG CÓ HÀNG"], key="position_mode", horizontal=True
    )
    if position_mode == "ĐANG CÓ HÀNG":
        qty = st.number_input("Số lượng đang nắm giữ", min_value=0, value=0, step=100, key="held_quantity")
        entry = st.number_input("Giá vốn (đồng/cp)", min_value=0.0, value=0.0, step=100.0, key="average_entry")
        stop_assumption = st.number_input("Stop Loss đang theo dõi (đồng/cp)", min_value=0.0, value=0.0, step=100.0, key="manual_stop")
    else:
        qty, entry, stop_assumption = 0, 0.0, 0.0

    with st.expander("Quản trị vốn & chi phí", expanded=False):
        nav = st.number_input("NAV mô phỏng (VND)", min_value=0, value=int(DEFAULT_CAPITAL_VND), step=5_000_000, key="nav_vnd")
        risk_pct = st.number_input("Rủi ro tối đa / giao dịch (%)", min_value=.1, max_value=5.0, value=1.0, step=.1, key="risk_pct")
        max_alloc = st.number_input("Tỷ trọng tối đa / mã (%)", min_value=1.0, max_value=100.0, value=20.0, step=1.0, key="max_alloc")
        lot = st.number_input("Bước lô", min_value=1, value=100, step=1, key="lot_size")
        exchange = st.selectbox("Sàn niêm yết", ["HOSE", "HNX", "UPCoM"], index=0, key="exchange_market")
        exchange_limit_pct = {"HOSE": 7.0, "HNX": 10.0, "UPCoM": 15.0}[exchange]
        fee = st.number_input("Phí mỗi chiều (%)", min_value=0.0, max_value=2.0, value=.15, step=.01, key="fee_pct")
        slip = st.number_input("Trượt giá mỗi chiều (%)", min_value=0.0, max_value=2.0, value=.10, step=.01, key="slippage_pct")
        tax_pct = st.number_input("Thuế bán (%)", min_value=0.0, max_value=2.0, value=float(DEFAULT_TAX_PCT), step=.01, key="tax_pct")


# Load the locally accumulated history. VNINDEX is stored in the same database,
# but is never price-scaled like an individual stock.
all_history = load_market_history(DATA_DB_PATH)
main_data = bench_data = sector_data = None
if not all_history.empty:
    symbol_key = all_history.symbol.astype(str).str.upper().str.replace("-", "", regex=False).str.replace("_", "", regex=False)
    bench_data = all_history.loc[symbol_key == "VNINDEX"].copy()
    main_data = all_history.loc[symbol_key != "VNINDEX"].copy()
    if main_data.empty:
        main_data = None
    if bench_data.empty:
        bench_data = None
sector_data = None

if main_data is not None:
    symbols = sorted(main_data.symbol.astype(str).unique().tolist())
    with st.sidebar:
        selected_symbol = st.selectbox("Chọn mã", symbols, index=0, key="selected_stock_symbol")
else:
    selected_symbol = None

if main_data is not None and selected_symbol:
    raw = main_data[main_data.symbol.astype(str) == selected_symbol].sort_values("date").reset_index(drop=True)
    data = raw.copy()  # canonical features are calculated exactly once in build_unified_snapshot
else:
    raw = data = None

# One central snapshot feeds all tabs. No tab performs its own analytical calculation.
shared_context = None
shared_context_error = None
swing_report = None
if data is not None and selected_symbol:
    try:
        sector_data = None
        sf_saved = st.session_state.get("sector_index_frame")
        if isinstance(sf_saved, pd.DataFrame) and selected_sector_name:
            sector_data = sf_saved[sf_saved.sector == selected_sector_name].copy()
            sector_data["symbol"] = selected_sector_name
            sector_data["open"] = sector_data["close"]
            sector_data["high"] = sector_data["close"]
            sector_data["low"] = sector_data["close"]
            sector_data["volume"] = (pd.to_numeric(sector_data["volume"], errors="coerce").fillna(0)
                                      if "volume" in sector_data.columns else 0.0)
        position_input = {"mode": "MANAGE_EXISTING" if position_mode == "ĐANG CÓ HÀNG" else "NEW_ENTRY",
                          "quantity": int(qty), "entry_price": float(entry), "stop_price": float(stop_assumption)}
        shared_context = build_unified_snapshot(
            selected_symbol, raw, universe=main_data, benchmark=bench_data, sector=sector_data,
            capital_vnd=float(nav), risk_pct=float(risk_pct), max_alloc_pct=float(max_alloc),
            lot_size=int(lot), position=position_input, fee_pct=float(fee),
            slippage_pct=float(slip), tax_pct=float(tax_pct))
        data = shared_context["enriched"]
        swing_report = shared_context["swing_report"]
    except Exception as exc:
        shared_context_error = str(exc)
if shared_context_error:
    st.error(f"Không thể tạo phiên phân tích thống nhất: {shared_context_error}")




# -------------------- V6.2.0: unified three-tab decision interface --------------------
def _safe_num(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None

st.markdown("### Phân tích tập trung · 3 tab")
st.caption("Một bộ dữ liệu chuẩn · Một bộ máy tính toán · Một snapshot dùng chung cho cả 3 tab. Chỉ dùng dữ liệu đã nạp; không tự tạo dữ liệu thị trường.")
main_tabs = st.tabs(["1 · TỔNG QUAN", "2 · DẤU CHÂN DÒNG TIỀN", "3 · CHIẾN LƯỢC"])

# TAB 1 — OVERVIEW
with main_tabs[0]:
    if data is None:
        st.info("Mở mục NẠP DỮ LIỆU ở thanh bên. Chỉ cần OHLCV; nên đưa VNINDEX vào cùng tệp để đánh giá bối cảnh thị trường.")
    elif swing_report is None:
        st.warning("Chưa tạo được báo cáo. Kiểm tra dữ liệu đầu vào và thông báo lỗi phía trên.")
    else:
        last = data.iloc[-1]
        score = swing_report.get("score")
        a, b, c, d = st.columns(4)
        a.metric("MÃ CỔ PHIẾU", selected_symbol)
        a.caption(f"Ngày dữ liệu cuối: {pd.to_datetime(last.date).strftime('%d/%m/%Y')}")
        b.metric("ĐIỂM TỔNG QUAN", "—" if score is None else f"{int(score)}/100")
        c.metric("PHA CỔ PHIẾU", (shared_context or {}).get("structure_analysis", {}).get("label", "CHƯA RÕ"))
        d.metric("GIÁ ĐÓNG CỬA", fmt(last.close, 0) + " đ")
        regime = (shared_context or {}).get("market_regime", {})
        market_label = market_status_label(regime)
        market_source = "VN-Index" if regime.get("benchmark") else ("Độ rộng nhóm mã (ước lượng)" if regime.get("source_type") == "BREADTH_PROXY" else "Chưa có benchmark")
        st.markdown("#### Bối cảnh thị trường chung")
        m1, m2, m3 = st.columns(3)
        m1.metric("TRẠNG THÁI THỊ TRƯỜNG", market_label)
        m2.metric("NGUỒN ĐÁNH GIÁ", market_source)
        m3.metric("ĐỘ TIN CẬY", regime.get("confidence", "THẤP"))
        for reason in regime.get("reasons", []): st.caption("• " + str(reason))
        if market_label == "CHƯA ĐỦ DỮ LIỆU":
            st.warning("Không có VN-Index hoặc độ rộng đủ đại diện. Không dùng riêng mã cổ phiếu đang phân tích để kết luận thị trường chung.")
        st.markdown("#### Đánh giá cơ hội")
        st.caption("Điểm Swing là điểm sàng lọc theo bộ quy tắc hiện tại, không phải xác suất thắng và không thay thế việc kiểm tra luận cứ.")
        if score is not None:
            st.progress(int(max(0, min(100, score))), text=f"Điểm sàng lọc: {int(score)}/100")
        st.markdown("#### Nhận định nhanh")
        st.write(f"**Mô hình hiện tại:** {swing_report.get('setup', 'Chưa xác định')}")
        st.write(f"**Kết luận:** {swing_report.get('conclusion', swing_report.get('decision', 'Chưa đủ dữ liệu'))}")
        positives = swing_report.get("positives", []) or []
        negatives = swing_report.get("negatives", []) or []
        left, right = st.columns(2)
        with left:
            st.markdown("**Điểm ủng hộ**")
            for item in positives[:4]: st.success(item)
            if not positives: st.caption("Chưa có luận cứ tích cực đủ điều kiện.")
        with right:
            st.markdown("**Rủi ro cần lưu ý**")
            for item in negatives[:4]: st.warning(item)
            if not negatives: st.caption("Chưa phát hiện theo bộ lọc hiện tại; không đồng nghĩa không có rủi ro.")
        with st.expander("Biểu đồ giá và MA20/MA50"):
            chart = data.tail(120)
            fig = go.Figure(data=[go.Candlestick(x=chart.date, open=chart.open, high=chart.high, low=chart.low, close=chart.close, name=selected_symbol)])
            for col_name in ("SMA20", "SMA50"):
                if col_name in chart.columns:
                    fig.add_trace(go.Scatter(x=chart.date, y=chart[col_name], mode="lines", name=col_name, line=dict(width=1)))
            fig.update_layout(height=390, margin=dict(l=5,r=5,t=20,b=5), xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        quality = shared_context.get("data_quality", {}) if shared_context else {}
        rs_m = (shared_context or {}).get("rs_market", {})
        rs_s = (shared_context or {}).get("rs_sector", {})
        r1,r2 = st.columns(2)
        r1.metric("SỨC MẠNH SO VỚI VN-INDEX · 20P", fmt(rs_m.get("relative_return_pct"),2) + "%")
        r2.metric("SỨC MẠNH SO VỚI NGÀNH · 20P", fmt(rs_s.get("relative_return_pct"),2) + "%")
        with st.expander("Phân tích tích lũy / hấp thụ / phân phối"):
            st.write((shared_context or {}).get("structure_analysis", {}).get("label", "Chưa đủ dữ liệu"))
            st.write("Độ tin cậy: " + str((shared_context or {}).get("structure_analysis", {}).get("confidence", "THẤP")))
            for ev in (shared_context or {}).get("structure_analysis", {}).get("evidence", []): st.write("• " + str(ev))
            for caution in (shared_context or {}).get("structure_analysis", {}).get("cautions", []): st.caption(caution)
        st.caption(f"Chất lượng dữ liệu: {quality.get('status', 'CHƯA RÕ')} · Số phiên: {len(raw)} · Snapshot ID: {shared_context.get('analysis_id','—')} · Engine: {shared_context.get('engine_version','—')}")

# TAB 2 — FLOW FOOTPRINT
with main_tabs[1]:
    if data is None:
        st.info("Nạp dữ liệu OHLCV để phân tích hành vi giá, khối lượng và các dấu hiệu bẫy giá.")
    else:
        flow = (shared_context or {}).get("flow_trap", {})
        fm = flow.get("metrics", {})
        st.markdown("#### Dấu chân dòng tiền từ giá và khối lượng")
        st.caption("Các kết quả dưới đây là proxy tính từ OHLCV; không xác định được danh tính tổ chức, khối ngoại hay tự doanh. Tất cả lấy từ cùng snapshot với tab Tổng quan và Chiến lược.")
        structure = (shared_context or {}).get("structure_analysis", {})
        st.markdown("#### Cấu trúc tích lũy – hấp thụ – phân phối")
        st.metric("ĐÁNH GIÁ CẤU TRÚC", structure.get("label", "CHƯA ĐỦ DỮ LIỆU"), help="Phân loại heuristic đa phiên từ CLV, cân bằng khối lượng và độ co hẹp biên độ; không xác định ý đồ người giao dịch.")
        st.caption(f"Độ tin cậy: {structure.get('confidence','THẤP')} · Điểm cấu trúc: {structure.get('score','—')}")
        for item in structure.get("evidence", []): st.write("• " + str(item))
        proxy_value = _safe_num(fm.get("obv_proxy_change_20"))
        p1, p2, p3 = st.columns(3)
        p1.metric("Xu hướng cung cầu", flow.get("flow_label", "CHƯA ĐỦ DỮ LIỆU"))
        p2.metric("Độ đồng thuận proxy", {"AGREE_BUY": "Nghiêng về cầu", "AGREE_SELL": "Nghiêng về cung", "DIVERGENT_OR_NEUTRAL": "Chưa đồng thuận"}.get(fm.get("proxy_agreement"), "Chưa đủ dữ liệu"))
        p3.metric("Rủi ro bẫy giá", flow.get("trap_label", "CHƯA ĐỦ DỮ LIỆU"))
        st.markdown("#### Dấu hiệu bất thường")
        anomaly_cols = st.columns(3)
        anomaly_cols[0].metric("Phiên khối lượng cao / đóng yếu", int(fm.get("weak_high_volume_sessions_5", 0) or 0), help="Cảnh báo áp lực cung; không tự nó chứng minh phân phối.")
        anomaly_cols[1].metric("CLV-volume proxy · 20 phiên", fmt(fm.get("flow_ratio_20"), 3))
        anomaly_cols[2].metric("Biến động giá · 20 phiên", fmt(fm.get("price_change_20_pct"), 2) + "%")
        if proxy_value is not None:
            st.caption(f"OBV proxy thay đổi 20 phiên: {proxy_value:,.0f}")
        trap_detail = flow.get("trap_analysis", {})
        with st.expander("Kiểm tra bẫy giá và breakout", expanded=True):
            st.caption("Sự kiện mới xuất hiện có thể chưa đủ phiên xác nhận. App phân biệt nghi vấn, chờ xác nhận, được xác nhận và bị vô hiệu.")
            trap_cols = st.columns(4)
            trap_cols[0].metric("Bull Trap xác nhận", fm.get("confirmed_bull_traps", 0))
            trap_cols[1].metric("Bull Trap chờ / mâu thuẫn", fm.get("pending_bull_traps", 0))
            trap_cols[2].metric("Breakout giữ được vùng", fm.get("confirmed_breakouts", 0))
            trap_cols[3].metric("Sự kiện mâu thuẫn", fm.get("conflicting_trap_events", 0))
            events = trap_detail.get("bull_trap", []) + trap_detail.get("bear_trap", []) + trap_detail.get("breakout", [])
            events = sorted(events, key=lambda e: e.get("date", ""), reverse=True)
            if events:
                display_events = [{
                    "Ngày": ev.get("date"),
                    "Loại": {"BULL_TRAP": "Nghi vấn bẫy tăng", "BEAR_TRAP": "Nghi vấn bẫy giảm", "BREAKOUT": "Breakout"}.get(ev.get("type"), ev.get("type")),
                    "Trạng thái": ev.get("label"),
                    "Mốc giá": round(float(ev.get("level", 0)), 3),
                    "Phiên theo dõi": ev.get("followup_sessions", 0),
                    "Diễn giải": ev.get("reason"),
                } for ev in events[:8]]
                st.dataframe(pd.DataFrame(display_events), use_container_width=True, hide_index=True)
            else:
                st.info("Chưa phát hiện mẫu hình bẫy hoặc breakout đủ điều kiện trong cửa sổ quét.")
            st.caption("Một tín hiệu chỉ được xác nhận khi diễn biến các phiên sau ủng hộ; không kết luận chỉ từ một cây nến.")
        st.markdown("#### Luận cứ phân tích")
        for item in flow.get("signals", [])[:8]: st.success(str(item))
        for item in flow.get("cautions", [])[:4]: st.warning("Lưu ý: " + str(item))
        st.info("Không có dữ liệu khối ngoại/tự doanh riêng trong đầu vào tối thiểu. Ứng dụng không suy diễn các dòng tiền này từ OHLCV.")

# TAB 3 — STRATEGY
with main_tabs[2]:
    if data is None:
        st.info("Nạp OHLCV ở thanh bên. Nếu đang mô phỏng vị thế, nhập giá vốn và số lượng tại khu vực VỊ THẾ MÔ PHỎNG.")
    elif swing_report is None:
        st.warning("Chưa đủ điều kiện tạo chiến lược.")
    else:
        st.markdown("#### Kết luận hành động")
        st.subheader(swing_report.get("conclusion", swing_report.get("decision", "CHƯA ĐỦ DỮ LIỆU")))
        st.caption("Kế hoạch tham chiếu theo quy tắc; không tự đặt lệnh. Nếu dữ liệu thiếu hoặc có điều kiện chặn, ưu tiên đứng ngoài/chờ xác nhận.")
        pos = swing_report.get("position", {})
        if pos.get("mode") == "MANAGE_EXISTING":
            st.markdown("### ĐANG CÓ HÀNG · Quản trị vị thế")
            p1,p2,p3,p4 = st.columns(4)
            p1.metric("Số lượng", f"{int(pos.get('quantity', 0)):,} cp")
            p2.metric("Giá vốn", fmt(pos.get("entry_price"), 0) + " đ")
            p3.metric("Giá hiện tại", fmt(pos.get("market_price"), 0) + " đ")
            p4.metric("Lãi/lỗ tạm tính", fmt(pos.get("pnl_pct_before_costs"), 2) + "%")
            p5,p6,p7 = st.columns(3)
            p5.metric("Mốc bảo vệ / SL", fmt(pos.get("reference_invalidation"), 0) + " đ")
            p6.metric("Mục tiêu 1", fmt(pos.get("target_1_reference"), 0) + " đ")
            p7.metric("Mục tiêu 2", fmt(pos.get("target_2_reference"), 0) + " đ")
            st.info(f"Đánh giá vị thế: {pos.get('status', 'CHƯA RÕ')} · Nguồn SL: {pos.get('stop_source', 'Tham chiếu')}")
            st.subheader(pos.get("management_action", "CHƯA ĐỦ DỮ LIỆU QUẢN TRỊ"))
            st.write("**Dời Stop Loss tham chiếu:** " + fmt(pos.get("trailing_stop_reference"), 0) + " đ")
            for stage in pos.get("management_stages", []):
                st.write(f"• **{stage.get('stage')}** — {stage.get('condition')}: {stage.get('action')}")
            if int(pos.get("quantity", 0)) <= 0 or not pos.get("entry_price"):
                st.warning("Nhập số lượng đang nắm giữ và giá vốn ở Sidebar để hệ thống đánh giá đúng vị thế.")
        else:
            st.markdown("### CHƯA CÓ HÀNG · Kế hoạch mở vị thế")
            plan = swing_report.get("plan", {})
            p1,p2,p3,p4 = st.columns(4)
            p1.metric("VÙNG THEO DÕI", f"{fmt(plan.get('entry_zone_low'), 0)} – {fmt(plan.get('entry_zone_high'), 0)} đ")
            p2.metric("GIÁ XÁC NHẬN", fmt(plan.get("reference_trigger"), 0) + " đ")
            p3.metric("MỨC VÔ HIỆU / SL", fmt(plan.get("invalidation"), 0) + " đ")
            p4.metric("MỤC TIÊU 1 / 2", f"{fmt(plan.get('target_1'), 0)} / {fmt(plan.get('target_2'), 0)} đ")
            st.write(f"**Tỷ lệ R:R:** TP1 {fmt(plan.get('rr_target_1'), 2)} : 1 · TP2 {fmt(plan.get('rr_target_2'), 2)} : 1")
            sizing = swing_report.get("sizing", {})
            q1,q2,q3 = st.columns(3)
            q1.metric("Khối lượng tham chiếu", f"{int(sizing.get('quantity', 0)):,} cp")
            q2.metric("Giá trị vị thế", f"{float(sizing.get('position_value', 0) or 0):,.0f} đ")
            q3.metric("Rủi ro mô phỏng", f"{float(sizing.get('risk_amount', 0) or 0):,.0f} đ")
            st.caption(f"Ngân sách rủi ro: {float(sizing.get('risk_budget', 0) or 0):,.0f} đ · Sizing: {sizing.get('status', 'CHƯA RÕ')}")
        blockers = swing_report.get("blockers", []) or []
        if blockers:
            st.markdown("#### Điều kiện chặn / chưa đạt")
            for item in blockers: st.warning(item)
        else:
            st.success("Không có điều kiện chặn theo bộ quy tắc hiện tại; vẫn cần xác minh dữ liệu và tuân thủ kế hoạch.")
        if swing_report.get("next_steps"):
            st.markdown("#### Việc cần làm tiếp theo")
            for item in swing_report.get("next_steps", []): st.write("• " + str(item))
        with st.expander("Luận cứ và dữ liệu kiểm tra"):
            st.write("**Điểm ủng hộ**")
            for item in swing_report.get("positives", []): st.write("• " + str(item))
            st.write("**Điểm bất lợi**")
            for item in swing_report.get("negatives", []): st.write("• " + str(item))
            st.json(swing_report.get("evidence", {}))
        with st.expander("TÓM TẮT DỄ HIỂU · VÌ SAO HỆ THỐNG ĐƯA RA KẾT LUẬN NÀY"):
            plain = build_beginner_summary(swing_report)
            st.markdown(f"**{plain.get('headline', 'CHƯA ĐỦ DỮ LIỆU')}**")
            st.write(plain.get("action", ""))
            for line in plain.get("plan_lines", []): st.write("• " + str(line))
            for warning in plain.get("warnings", []): st.caption("Lưu ý: " + str(warning))

        st.download_button(
            "TẢI BÁO CÁO CHIẾN LƯỢC JSON",
            data=json.dumps(swing_report, ensure_ascii=False, indent=2, default=str).encode("utf-8-sig"),
            file_name=f"{selected_symbol}_swing_strategy_v6.2.0.json",
            mime="application/json", key="download_swing_strategy_v620"
        )
        with st.expander("KIỂM CHỨNG LỊCH SỬ · MÔ PHỎNG 3–10 PHIÊN"):
            st.caption(f"Mô phỏng dùng benchmark VN-Index, T+2, phí/thuế/trượt giá và biên độ {exchange} {exchange_limit_pct:.0f}%. Chỉ tính lệnh đã đóng vào thống kê hiệu suất.")
            if bench_data is None:
                st.warning("Cần tải VN-Index đồng bộ ở Sidebar trước khi chạy kiểm chứng. Không dùng riêng mã cổ phiếu làm benchmark.")
            elif st.button("CHẠY WALK-FORWARD + KIỂM ĐỊNH NGOÀI MẪU", key="run_walkforward_backtest"):
                with st.spinner("Đang chạy kiểm định tuần tự theo thời điểm, tách riêng giai đoạn ngoài mẫu..."):
                    split_idx = int(len(raw) * 0.70)
                    train_raw = raw.iloc[:split_idx].copy()
                    # Holdout starts after 70% history; prior rows remain available only for indicator warm-up.
                    oos = backtest_plan(raw, benchmark=bench_data, capital=float(nav), risk_pct=float(risk_pct),
                        max_alloc_pct=float(max_alloc), fee_pct=float(fee), tax_pct=float(tax_pct),
                        slip_pct=float(slip), limit_pct=float(exchange_limit_pct), max_hold=10,
                        test_start_index=split_idx, sector=sector_data)
                if "error" in oos:
                    st.error(oos["error"])
                else:
                    st.markdown("**KẾT QUẢ NGOÀI MẪU · 30% DỮ LIỆU CUỐI**")
                    b1,b2,b3,b4 = st.columns(4)
                    b1.metric("LỆNH ĐÃ ĐÓNG", oos["closed_trades"])
                    b2.metric("TỶ LỆ THẮNG", fmt(oos.get("win_rate_pct"), 1) + "%")
                    b3.metric("EXPECTANCY", fmt(oos.get("expectancy_pct"), 2) + "%")
                    b4.metric("PROFIT FACTOR", fmt(oos.get("profit_factor"), 2))
                    b5,b6,b7 = st.columns(3)
                    b5.metric("LỢI NHUẬN MÔ PHỎNG", fmt(oos.get("total_return_pct"), 2) + "%")
                    b6.metric("DRAWDOWN GIAO DỊCH", fmt(oos.get("max_drawdown_pct"), 2) + "%")
                    b7.metric("VỊ THẾ CÒN MỞ", int(oos.get("open_positions", 0)))
                    st.caption(f"Giai đoạn ngoài mẫu bắt đầu tại phiên {split_idx+1}/{len(raw)}. Số phiên kiểm định: {oos.get('sessions_tested')}.")
                    if oos.get("enough_sample"):
                        st.success(oos["note"])
                    else:
                        st.warning("Mẫu ngoài mẫu dưới 30 lệnh: chưa đủ cơ sở kết luận hiệu quả. " + oos["note"])
                    if oos.get("trades"):
                        st.dataframe(pd.DataFrame(oos["trades"]).tail(30), use_container_width=True, hide_index=True)
                    st.info("Đây là holdout ngoài mẫu với quy tắc cố định và tín hiệu tính tại từng thời điểm; chưa phải bằng chứng hiệu quả tương lai. Không tối ưu tham số trên chính tập kiểm định này.")

st.divider()
st.caption("V6.2.0 · 3 tab: Tổng quan / Dấu chân dòng tiền / Chiến lược · Khung Swing 3–10 phiên. Dữ liệu khối ngoại, tự doanh và lệnh lớn chỉ được dùng khi người dùng cung cấp file có nguồn; OHLCV là proxy, không xác định danh tính tổ chức. Bộ quy tắc cần kiểm định ngoài mẫu trên dữ liệu thật trước khi đánh giá hiệu quả.")
