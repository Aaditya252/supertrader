"use client";
import { useEffect, useRef } from "react";
import { createChart, CandlestickSeries, HistogramSeries, createSeriesMarkers, ColorType, type UTCTimestamp } from "lightweight-charts";
import type { Candle, Signal } from "@/lib/types";

export default function MarketChart({ candles, signal, history }: {candles:Candle[];signal:Signal|null;history:Signal[]}) {
  const container = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!container.current || !candles.length) return;
    const chart = createChart(container.current, {
      autoSize:true, height:392,
      layout:{background:{type:ColorType.Solid,color:"#11151c"},textColor:"#818b9b",fontFamily:"monospace",fontSize:11,attributionLogo:true},
      grid:{vertLines:{color:"#1b212a"},horzLines:{color:"#1b212a"}},
      rightPriceScale:{borderColor:"#252d39",scaleMargins:{top:.08,bottom:.22}},
      timeScale:{borderColor:"#252d39",timeVisible:true,secondsVisible:false},
      crosshair:{vertLine:{color:"#52637a"},horzLine:{color:"#52637a"}}
    });
    const series = chart.addSeries(CandlestickSeries,{upColor:"#68ddba",downColor:"#e67786",wickUpColor:"#68ddba",wickDownColor:"#e67786",borderVisible:false});
    series.setData(candles.map(c=>({...c,time:c.time as UTCTimestamp})));
    const volume = chart.addSeries(HistogramSeries,{priceFormat:{type:"volume"},priceScaleId:"volume"});
    volume.priceScale().applyOptions({scaleMargins:{top:.83,bottom:0},visible:false});
    volume.setData(candles.map(c=>({time:c.time as UTCTimestamp,value:c.volume,color:c.close>=c.open?"#244a42":"#4d303b"})));
    const valid = new Set(candles.map(c=>c.time));
    const unique = new Map<number,Signal>();
    history.filter(s=>s.action!=="HOLD" && s.decision_time).forEach(s=>unique.set(Math.floor(s.decision_time!/1000)-3600,s));
    createSeriesMarkers(series, [...unique].filter(([t])=>valid.has(t)).sort(([a],[b])=>a-b).map(([t,s])=>({time:t as UTCTimestamp,position:s.action==="BUY"?"belowBar" as const:"aboveBar" as const,color:s.action==="BUY"?"#68ddba":"#e67786",shape:s.action==="BUY"?"arrowUp" as const:"arrowDown" as const,text:s.action})));
    if (signal?.stop_loss) series.createPriceLine({price:signal.stop_loss,color:"#e67786",lineWidth:1,lineStyle:2,axisLabelVisible:true,title:"STOP"});
    if (signal?.take_profit) series.createPriceLine({price:signal.take_profit,color:"#68ddba",lineWidth:1,lineStyle:2,axisLabelVisible:true,title:"TARGET"});
    chart.timeScale().setVisibleLogicalRange({from:Math.max(0,candles.length-110),to:candles.length+3});
    return ()=>chart.remove();
  },[candles, signal?.stop_loss, signal?.take_profit, history]);
  return <div className="market-chart" ref={container}>{!candles.length && <div className="chart-empty"><div className="skeleton chart-skeleton"/><span>Waiting for verified exchange candles</span></div>}</div>;
}
