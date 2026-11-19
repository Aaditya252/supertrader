"use client";
import { useState } from "react";
import { FlowAsset, moveAt, SESSION_STEPS, sessionTime, signed } from "@/lib/market-flow";

type Props={assets:FlowAsset[];step:number;selected:string|null;onSelect:(a:FlowAsset)=>void;showLabels:boolean;compact?:boolean};
const GREEN="#119b70",RED="#e25c77";
export default function FlowCanvas({assets,step,selected,onSelect,showLabels,compact=false}:Props){
  const [hover,setHover]=useState<string|null>(null);
  const W=1240,H=compact?250:540,left=57,right=1067,top=35,bottom=H-47;
  const mid=(top+bottom)/2;
  const bound=Math.max(3,...assets.flatMap(a=>a.series.map(Math.abs)))*1.12;
  const y=(n:number)=>mid-n/bound*(bottom-top)/2;
  const x=(i:number)=>left+i/SESSION_STEPS*(right-left);
  const ranked=[...assets].sort((a,b)=>moveAt(b,step)-moveAt(a,step));
  const focus=hover??selected;
  const hovered=assets.find(a=>a.symbol===hover);
  return <div className={`mf-canvas ${compact?"mf-compact-canvas":""}`}>
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Session price trajectories for ${assets.length} sample stocks through ${sessionTime(step)}. Select an endpoint to inspect a stock.`}>
      <defs>
        <linearGradient id={compact?"flow-up-small":"flow-up"} x1="0" x2="0" y1="1" y2="0"><stop offset="0" stopColor="#119b70" stopOpacity="0"/><stop offset="1" stopColor="#119b70" stopOpacity=".065"/></linearGradient>
        <linearGradient id={compact?"flow-down-small":"flow-down"} x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#e25c77" stopOpacity="0"/><stop offset="1" stopColor="#e25c77" stopOpacity=".045"/></linearGradient>
      </defs>
      <rect x={left} y={top} width={right-left} height={mid-top} fill={`url(#${compact?"flow-up-small":"flow-up"})`}/>
      <rect x={left} y={mid} width={right-left} height={bottom-mid} fill={`url(#${compact?"flow-down-small":"flow-down"})`}/>
      {[-4,-2,0,2,4].filter(n=>Math.abs(n)<bound).map(n=><g key={n}><line x1={left} x2={right} y1={y(n)} y2={y(n)} stroke={n===0?"#aab6bb":"#e1e8e8"} strokeDasharray={n===0?undefined:"3 5"}/><text x={left-13} y={y(n)+4} textAnchor="end" className="mf-axis">{n>0?"+":""}{n}%</text></g>)}
      {[0,9,21,33,45,57,75].map(i=><g key={i}><line x1={x(i)} x2={x(i)} y1={top} y2={bottom} stroke="#e9eeee" strokeDasharray="3 5"/><text x={x(i)} y={H-15} textAnchor="middle" className="mf-axis">{sessionTime(i)}</text></g>)}
      {assets.map((a,i)=>{
        const end=moveAt(a,step),color=end>=0?GREEN:RED,isFocus=focus===a.symbol;
        const d=a.series.slice(0,step+1).map((v,k)=>`${k?"L":"M"}${x(k).toFixed(2)},${y(v).toFixed(2)}`).join(" ");
        const isLeader=ranked.indexOf(a)<3||ranked.indexOf(a)>=ranked.length-3;
        return <g key={a.symbol} opacity={focus&&!isFocus?0.12:1} className="mf-trajectory">
          <path d={d} fill="none" stroke={color} strokeWidth={isFocus?2.8:isLeader?1.7:1.05} opacity={isFocus?1:isLeader?.75:.32+(i%4)*.08} vectorEffect="non-scaling-stroke"/>
          {!compact&&step>4&&<circle r={isFocus?3.2:1.8} fill={color} opacity=".75"><animateMotion dur={`${4+i%6}s`} repeatCount="indefinite" path={d}/></circle>}
          <g role="button" tabIndex={0} aria-label={`Inspect ${a.symbol}, ${signed(end)}`} onClick={()=>onSelect(a)} onKeyDown={e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();onSelect(a);}}} onMouseEnter={()=>setHover(a.symbol)} onMouseLeave={()=>setHover(null)} onFocus={()=>setHover(a.symbol)} onBlur={()=>setHover(null)} className="mf-endpoint">
            <circle cx={x(step)} cy={y(end)} r={10} fill="transparent"/>
            <circle cx={x(step)} cy={y(end)} r={isFocus?5:2.7} fill={color} stroke="#fff" strokeWidth={isFocus?2:.6}/>
            {(isFocus||(showLabels&&(ranked.indexOf(a)<3||ranked.indexOf(a)>=ranked.length-3)))&&<text x={x(step)+11} y={y(end)+4} fill={color} className="mf-end-label">{a.symbol}</text>}
          </g>
        </g>;
      })}
      <circle cx={left} cy={mid} r="4.5" fill="#52637b" stroke="white" strokeWidth="2"/>
      {step<SESSION_STEPS&&<line x1={x(step)} x2={x(step)} y1={top} y2={bottom} stroke="#778597" strokeDasharray="4 4"/>}
      {!compact&&<><text x={right+30} y={top+5} fill={GREEN} className="mf-chart-tag">↑ LEADING</text><text x={right+30} y={bottom-5} fill={RED} className="mf-chart-tag">↓ LAGGING</text><text x={left+12} y={mid-10} className="mf-axis">SESSION OPEN</text></>}
    </svg>
    {hovered&&<div className="mf-chart-tooltip" role="status"><b>{hovered.symbol}</b><span>{hovered.sector}</span><strong className={moveAt(hovered,step)>=0?"mf-positive":"mf-negative"}>{signed(moveAt(hovered,step))}</strong><small>Select to inspect · {sessionTime(step)} IST</small></div>}
  </div>;
}
