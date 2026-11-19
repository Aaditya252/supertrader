"use client";

import {useCallback,useEffect,useState} from "react";
import {Activity,Database,RefreshCw,ShieldCheck,Wifi,WifiOff} from "lucide-react";
import type {OrderFlowSnapshot,OrderFlowWindow} from "@/lib/types";
import "@/app/orderflow.css";

const API=process.env.NEXT_PUBLIC_API_URL??"http://localhost:8000";
const usd=(value:number|null|undefined)=>value==null?"—":value.toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2});
const compactUsd=(value:number)=>value>=1_000_000?`$${(value/1_000_000).toFixed(2)}m`:value>=10_000?`$${(value/1_000).toFixed(1)}k`:`$${value.toLocaleString("en-US",{maximumFractionDigits:0})}`;
const signedUsd=(value:number|null|undefined)=>value==null?"—":`${value>=0?"+":"−"}${compactUsd(Math.abs(value))}`;
const pct=(value:number|null|undefined)=>value==null?"—":`${(value*100).toFixed(1)}%`;
const clock=(value:number)=>new Date(value).toLocaleTimeString(undefined,{hour:"2-digit",minute:"2-digit",second:"2-digit"});

function WindowCard({label,data}:{label:string;data?:OrderFlowWindow}){
  return <div className="of-window-card">
    <span>{label} TAKER FLOW</span>
    <strong className={(data?.net_taker_delta_usdt??0)>=0?"of-buy-text":"of-sell-text"}>{signedUsd(data?.net_taker_delta_usdt)}</strong>
    <div><span>Buy {compactUsd(data?.buy_notional_usdt??0)}</span><span>Sell {compactUsd(data?.sell_notional_usdt??0)}</span></div>
    <small>{data?.aggregate_trade_count.toLocaleString()??"0"} Binance aggregate trades · imbalance {pct(data?.imbalance)}</small>
  </div>;
}

export default function OrderFlowPanel(){
  const [snapshot,setSnapshot]=useState<OrderFlowSnapshot|null>(null);
  const [error,setError]=useState<string|null>(null);
  const [loading,setLoading]=useState(true);
  const refresh=useCallback(async()=>{
    try{
      const response=await fetch(`${API}/orderflow`,{cache:"no-store",signal:AbortSignal.timeout(6000)});
      if(!response.ok)throw new Error(`HTTP ${response.status}`);
      setSnapshot(await response.json());setError(null);
    }catch{
      setError("Live order-flow endpoint unavailable. Check the local backend and exchange feed.");
    }finally{setLoading(false);}
  },[]);
  useEffect(()=>{void refresh();const timer=setInterval(()=>void refresh(),2000);return()=>clearInterval(timer);},[refresh]);

  const book=snapshot?.book;
  const flow=snapshot?.flow_windows;
  const buckets=snapshot?.depth_map.buckets??[];
  const largestDepth=Math.max(1,...buckets.flatMap(row=>[row.bid_notional_usdt,row.ask_notional_usdt]));
  const age=snapshot?.feed_age_seconds;
  const feedLive=Boolean(snapshot?.connected&&age!=null&&age<15);
  const bookReady=Boolean(book?.book_valid);

  return <div className="of-page">
    <div className="validation-banner of-banner"><Activity size={22}/><div><b>Live order flow · {snapshot?.symbol??"SOLUSDT"}</b><p>Anonymous Binance taker flow and sequenced spot depth, polled every 2 seconds.</p></div><span className={`badge ${feedLive?"green":"amber"}`}>{feedLive?"LIVE FEED":"WAITING FOR FEED"}</span></div>
    {error&&<div className="error-banner" role="alert">{error}<button onClick={()=>void refresh()}>Retry</button></div>}
    <div className="of-disclosure"><ShieldCheck size={15}/><span>BUY / SELL marks the aggressive taker side. Binance public data does not identify the buyer, seller, account, or other venue.</span><span>{bookReady?"SEQUENCED BOOK":"BOOK SYNCING"}</span></div>

    <div className="of-stat-grid">
      <div className="stat"><span>BINANCE SPOT MID</span><strong>${usd(book?.mid)}</strong><small>{snapshot?.venue??"Binance Spot"} · {snapshot?.symbol??"SOLUSDT"}</small></div>
      <div className="stat"><span>SPREAD / MICROPRICE</span><strong>{book?.spread_bps?.toFixed(2)??"—"}<em> bps</em></strong><small>μ ${usd(book?.microprice)} · offset {book?.microprice_offset_bps?.toFixed(2)??"—"} bps</small></div>
      <div className="stat"><span>BEST-20 DEPTH IMBALANCE</span><strong className={(book?.imbalance??0)>=0?"of-buy-text":"of-sell-text"}>{pct(book?.imbalance)}</strong><small>Bid versus ask notional · top 20 levels</small></div>
      <div className="stat"><span>RELATIVE LARGE-PRINT LEVEL</span><strong>{snapshot?.large_print_threshold_usdt==null?"Warming up":compactUsd(snapshot.large_print_threshold_usdt)}</strong><small>95th percentile · {snapshot?.large_print_sample_count??0} recent prints</small></div>
    </div>

    <div className="of-window-grid">
      <WindowCard label="1 minute" data={flow?.["1m"]}/>
      <WindowCard label="5 minute" data={flow?.["5m"]}/>
      <WindowCard label="15 minute" data={flow?.["15m"]}/>
    </div>

    <div className="of-main-grid">
      <section className="panel of-depth-panel">
        <div className="panel-heading"><div><h2><Database size={16}/>Live depth heatmap</h2><p className="of-subheading">Current sequenced bids and asks, grouped into price bands within ±{snapshot?.depth_map.range_bps??50} bps of mid.</p></div><span className={`badge ${bookReady?"green":"amber"}`}>{snapshot?.depth_map.status??"Waiting for book"}</span></div>
        {buckets.length?<>
          <div className="of-depth-legend"><span className="of-buy-text">BID LIQUIDITY · USDT</span><span>OFFSET FROM MID</span><span className="of-sell-text">ASK LIQUIDITY · USDT</span></div>
          <div className="of-depth-list">{buckets.map(row=><div className="of-depth-row" key={row.distance_bps}>
            <div className="of-depth-bid"><span>{compactUsd(row.bid_notional_usdt)}</span><i><b style={{width:`${row.bid_notional_usdt/largestDepth*100}%`}}/></i></div>
            <span className="of-depth-offset">±{row.distance_bps.toFixed(1)} bps</span>
            <div className="of-depth-ask"><i><b style={{width:`${row.ask_notional_usdt/largestDepth*100}%`}}/></i><span>{compactUsd(row.ask_notional_usdt)}</span></div>
          </div>)}</div>
          <p className="footnote">{snapshot?.depth_map.basis}. Values are visible aggregate quantities; displayed liquidity can be canceled and is not a confirmed order block.</p>
        </>:<div className="empty"><Database size={22}/><p>{loading?"Connecting to the sequenced book…":"No synchronized order book yet."}</p><small>Heatmap values appear after the Binance snapshot is bridged to live depth updates.</small></div>}
      </section>

      <section className="panel of-tape-panel">
        <div className="panel-heading"><div><h2><Activity size={16}/>Recent trade tape</h2><p className="of-subheading">Latest anonymous aggregate trades from this venue.</p></div><span className="badge neutral">{snapshot?.recent_trades.length??0} PRINTS</span></div>
        <div className="of-tape-head"><span>SIDE · TIME</span><span>PRICE</span><span>SIZE</span><span>NOTIONAL</span></div>
        <div className="of-tape-list">{snapshot?.recent_trades.map(trade=><div className={`of-tape-row ${trade.side==="BUY"?"is-buy":"is-sell"}`} key={trade.aggregate_id}>
          <div><b className={trade.side==="BUY"?"of-buy-text":"of-sell-text"}>{trade.side}</b><small>{clock(trade.time)}</small></div>
          <span>${usd(trade.price)}</span><span>{trade.quantity.toLocaleString("en-US",{maximumFractionDigits:4})}</span>
          <strong>{compactUsd(trade.notional_usdt)}{trade.relative_large&&<i className="of-large-tag">LARGE*</i>}</strong>
        </div>)}{!snapshot?.recent_trades.length&&<div className="empty compact">Recent trades will appear when the live aggregate-trade stream is connected.</div>}</div>
        <p className="footnote">* “Large” means at or above the rolling 95th percentile of recent aggregate-trade notional. It is a relative print-size flag, not a whale or account label.</p>
      </section>
    </div>

    <div className="of-footer"><span>{feedLive?<Wifi size={13}/>:<WifiOff size={13}/>} Feed age {age==null?"—":`${age.toFixed(1)}s`}</span><span>{snapshot?.book?.sequence_id?`Book sequence ${snapshot.book.sequence_id.toLocaleString()}`:"Book sequence unavailable"}</span><span>Snapshot {snapshot?.generated_at?clock(snapshot.generated_at):"—"}</span><button className="icon-button" aria-label="Refresh order flow" onClick={()=>void refresh()}><RefreshCw size={15}/></button></div>
  </div>;
}
