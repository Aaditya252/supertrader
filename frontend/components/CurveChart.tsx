"use client";
import type { Curve } from "@/lib/types";
export default function CurveChart({a,b,drawdown=false}:{a:Curve[];b?:Curve[];drawdown?:boolean}) {
  if (!a.length) return <div className="empty">No simulation results available yet.</div>;
  const all = [...a,...(b??[])].map(p=>drawdown?p.drawdown:p.equity);
  const low=Math.min(...all),high=Math.max(...all),range=high-low || .01;
  const path=(points:Curve[])=>points.map((p,i)=>`${38+i/Math.max(points.length-1,1)*900},${25+(high-(drawdown?p.drawdown:p.equity))/range*190}`).join(" ");
  return <div className="curve"><svg viewBox="0 0 970 250" role="img" aria-label={drawdown?"Backtest drawdown curve":"Equity comparison curve"}>
    {[0,1,2,3].map(i=><g key={i}><line x1="38" x2="940" y1={25+i*190/3} y2={25+i*190/3} stroke="#252c37"/><text x="0" y={29+i*190/3} fill="#8d97a7" fontSize="10">{(high-i*range/3).toFixed(2)}</text></g>)}
    {b&&<polyline points={path(b)} fill="none" stroke="#818aaa" strokeWidth="2"/>}
    <polyline points={path(a)} fill="none" stroke={drawdown?"#e67786":"#68ddba"} strokeWidth="2.5"/>
    <text x="38" y="242" fill="#8d97a7" fontSize="11">{new Date(a[0].time).toLocaleDateString()}</text><text x="860" y="242" fill="#8d97a7" fontSize="11">{new Date(a[a.length-1].time).toLocaleDateString()}</text>
  </svg></div>;
}
