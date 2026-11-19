"use client";
import {useEffect,useMemo,useRef,useState} from "react";
import Link from "next/link";
import {Activity,ArrowDownRight,ArrowLeft,ArrowRight,ArrowUpRight,Check,ChevronDown,ChevronRight,Download,Expand,GitBranch,Globe2,Info,Layers3,LayoutGrid,Menu,Pause,Play,RotateCcw,Search,SlidersHorizontal,Star,TrendingDown,TrendingUp,Waypoints,X,Zap} from "lucide-react";
import {FlowAsset,FlowStation,moveAt,priceAt,sectorStats,SESSION_STEPS,sessionTime,signed} from "@/lib/market-flow";
import FlowCanvas from "./FlowCanvas";
import {getMarketFlowSnapshot} from "@/lib/api/market-flow";
const DEMO_ASSETS = getMarketFlowSnapshot().assets;

const STATIONS=[{id:"market",name:"Market overview",short:"Market",icon:Globe2},{id:"flow",name:"Session flow",short:"Flow",icon:GitBranch},{id:"bullish",name:"Bullish board",short:"Bullish",icon:TrendingUp},{id:"bearish",name:"Bearish board",short:"Bearish",icon:TrendingDown}] as const;
const rupees=(n:number)=>`₹${n.toLocaleString("en-IN",{minimumFractionDigits:2,maximumFractionDigits:2})}`;

function Sparkline({asset,step,large=false}:{asset:FlowAsset;step:number;large?:boolean}){
  const values=asset.series.slice(0,step+1),lo=Math.min(0,...values)-.2,hi=Math.max(.1,...values)+.2;
  const d=values.map((v,i)=>`${i?"L":"M"}${i/SESSION_STEPS*240},${65-(v-lo)/(hi-lo)*56}`).join(" ");
  const positive=moveAt(asset,step)>=0;
  return <svg className={large?"mf-spark-large":"mf-spark"} viewBox="0 0 240 75" aria-hidden="true"><path d={`${d} L${step/SESSION_STEPS*240},75 L0,75 Z`} fill={positive?"#119b7010":"#e25c7710"}/><path d={d} fill="none" stroke={positive?"#119b70":"#e25c77"} strokeWidth="2" vectorEffect="non-scaling-stroke"/></svg>;
}

export default function MarketFlow(){
  const [station,setStation]=useState<FlowStation>("flow");
  const [sector,setSector]=useState<string|null>(null),[query,setQuery]=useState("");
  const [step,setStep]=useState(SESSION_STEPS),[playing,setPlaying]=useState(false),[labels,setLabels]=useState(true);
  const [selected,setSelected]=useState<FlowAsset|null>(null),[search,setSearch]=useState(false),[info,setInfo]=useState(false),[menu,setMenu]=useState(false);
  const [watchlist,setWatchlist]=useState<string[]>([]),[watchOnly,setWatchOnly]=useState(false),[ready,setReady]=useState(false);
  const [expanded,setExpanded]=useState(false);
  const searchInput=useRef<HTMLInputElement>(null),closeDetail=useRef<HTMLButtonElement>(null),dialogRef=useRef<HTMLDivElement>(null);
  useEffect(()=>{try{const stored=JSON.parse(localStorage.getItem("supertrader.flow.watchlist")??"[]");if(Array.isArray(stored))setWatchlist(stored.filter(x=>typeof x==="string"));}catch{/* Ignore invalid browser-local preferences. */}setReady(true);},[]);
  useEffect(()=>{if(ready)try{localStorage.setItem("supertrader.flow.watchlist",JSON.stringify(watchlist));}catch{/* Storage may be disabled. */}},[watchlist,ready]);
  useEffect(()=>{
    const openSearch=(e:KeyboardEvent)=>{
      const target=e.target as HTMLElement;
      if(e.key==="/"&&!selected&&!search&&!info&&!target.matches("input,textarea,[contenteditable=true]")){
        e.preventDefault();setQuery("");setSearch(true);
      }
    };
    document.addEventListener("keydown",openSearch);return()=>document.removeEventListener("keydown",openSearch);
  },[selected,search,info]);
  useEffect(()=>{if(!playing)return;const timer=setInterval(()=>setStep(s=>{if(s>=SESSION_STEPS){setPlaying(false);return SESSION_STEPS;}return s+1;}),130);return()=>clearInterval(timer);},[playing]);
  useEffect(()=>{
    if(!selected&&!search&&!info)return;
    const previous=document.activeElement as HTMLElement|null;
    const previousOverflow=document.body.style.overflow;
    document.body.style.overflow="hidden";
    if(search)searchInput.current?.focus();else closeDetail.current?.focus();
    const handler=(e:KeyboardEvent)=>{
      if(e.key==="Escape"){setSelected(null);setSearch(false);setInfo(false);}
      if(e.key==="Tab"){
        const nodes=dialogRef.current?.querySelectorAll<HTMLElement>('button:not([disabled]), input, a[href], [tabindex="0"]');
        if(!nodes?.length)return;const first=nodes[0],last=nodes[nodes.length-1];
        if(e.shiftKey&&document.activeElement===first){e.preventDefault();last.focus();}
        else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first.focus();}
      }
    };
    document.addEventListener("keydown",handler);return()=>{document.removeEventListener("keydown",handler);document.body.style.overflow=previousOverflow;previous?.focus();};
  },[selected,search,info]);
  const sectors=useMemo(()=>sectorStats(DEMO_ASSETS,step),[step]);
  const assets=useMemo(()=>DEMO_ASSETS.filter(a=>(!sector||a.sector===sector)&&(!watchOnly||watchlist.includes(a.symbol))),[sector,watchOnly,watchlist]);
  const ranked=useMemo(()=>[...assets].sort((a,b)=>moveAt(b,step)-moveAt(a,step)),[assets,step]);
  const winners=ranked.filter(a=>moveAt(a,step)>0),losers=[...ranked].reverse().filter(a=>moveAt(a,step)<0);
  const flat=assets.length-winners.length-losers.length;
  const average=assets.length?assets.reduce((s,a)=>s+moveAt(a,step),0)/assets.length:0;
  const searchResults=DEMO_ASSETS.filter(a=>`${a.symbol} ${a.name} ${a.sector}`.toLowerCase().includes(query.toLowerCase()));
  const toggleWatch=(symbol:string)=>setWatchlist(w=>w.includes(symbol)?w.filter(x=>x!==symbol):[...w,symbol]);
  const startReplay=()=>{if(playing){setPlaying(false);return;}if(step===SESSION_STEPS)setStep(0);setPlaying(true);};
  const inspect=(a:FlowAsset)=>{setPlaying(false);setSearch(false);setSelected(a);};
  function download(){
    const rows=["symbol,sector,session_time_IST,sample_price,sample_change_percent,data_mode",...assets.map(a=>`${a.symbol},${a.sector},${sessionTime(step)},${priceAt(a,step).toFixed(2)},${moveAt(a,step).toFixed(4)},synthetic_design_preview`)];
    const url=URL.createObjectURL(new Blob([rows.join("\n")],{type:"text/csv;charset=utf-8;"}));const link=document.createElement("a");link.href=url;link.download="supertrader-market-flow-sample.csv";link.click();URL.revokeObjectURL(url);
  }
  const detailPositive=selected?moveAt(selected,step)>=0:true;
  return <div className={`mf-app ${expanded?"mf-expanded":""}`}>
    <aside className={`mf-sidebar ${menu?"mf-mobile-open":""}`}>
      <Link href="/" className="mf-brand"><span className="mf-brand-icon"><Zap size={19} fill="currentColor"/></span>supertrader<span className="mf-brand-period">.</span></Link>
      <div className="mf-workspace"><span className="mf-workspace-avatar">S</span><div><b>Personal workspace</b><small>Research & discovery</small></div><ChevronDown size={13}/></div>
      <p className="mf-nav-caption">WORKSPACE</p>
      <Link href="/" className="mf-nav-link"><LayoutGrid size={17}/>Research terminal<ArrowUpRight size={13}/></Link>
      <div className="mf-nav-link mf-current"><Waypoints size={17}/>Market Flow<span className="mf-new">NEW</span></div>
      <div className="mf-subnav">{STATIONS.map(s=><button key={s.id} className={station===s.id?"is-active":""} onClick={()=>{setStation(s.id);setMenu(false);}}><s.icon size={15}/>{s.name}{station===s.id&&<i/>}</button>)}</div>
      <p className="mf-nav-caption">YOUR COLLECTION</p>
      <button className={`mf-nav-link ${watchOnly?"mf-current":""}`} onClick={()=>{setWatchOnly(!watchOnly);setMenu(false);}}><Star size={17}/>Watchlist<span className="mf-count">{watchlist.length}</span></button>
      <button className="mf-nav-link" onClick={()=>setInfo(true)}><Layers3 size={17}/>How to read the flow</button>
      <div className="mf-sidebar-bottom"><div className="mf-build-note"><span className="mf-mini-icon"><GitBranch size={19}/></span><b>A wider market perspective.</b><p>Follow the session. Find the leaders. Inspect the details.</p><button onClick={()=>setInfo(true)}>Explore the methodology<ArrowUpRight size={13}/></button></div><Link href="/" className="mf-back"><ArrowLeft size={15}/>Back to your terminal</Link><div className="mf-profile"><span>AD</span><div><b>Personal workspace</b><small>Supertrader Research</small></div></div></div>
    </aside>
    {menu&&<button className="mf-menu-scrim" aria-label="Close navigation" onClick={()=>setMenu(false)}/>}
    <main className="mf-main">
      <header className="mf-topbar"><div className="mf-breadcrumb"><button className="mf-mobile-menu" aria-label="Open navigation" onClick={()=>setMenu(!menu)}><Menu size={19}/></button><span>Workspace</span><ChevronRight size={13}/><b>Market Flow</b></div><div className="mf-topbar-actions"><span className="mf-preview-dot"/>Design preview<span className="mf-topbar-divider"/><button onClick={()=>setInfo(true)} aria-label="About this data"><Info size={17}/></button><span className="mf-top-avatar">AD</span></div></header>
      <div className="mf-content">
        <div className="mf-page-title"><div><div className="mf-eyebrow"><span/>THE MARKET, CONNECTED</div><h1>See the move. <span>Follow the flow.</span></h1><p>One session. Every sector. A clearer view of what’s moving.</p></div><button className="mf-universe" onClick={()=>setInfo(true)}><span className="mf-india">IN</span><div><b>India equities</b><small>Curated sample universe</small></div><ChevronDown size={15}/></button></div>
        <div className="mf-data-note"><span><Info size={14}/><b>Interactive design preview</b><span className="mf-note-detail">· Synthetic prices and trajectories. No live exchange feed connected.</span></span><button onClick={()=>setInfo(true)}>Data & methodology<ArrowUpRight size={13}/></button></div>
        <div className="mf-summary">
          <div><span>STOCKS IN VIEW</span><strong>{assets.length}<small>/ {DEMO_ASSETS.length}</small></strong><p>{sector??(watchOnly?"Your watchlist":"Across 12 sectors")}</p></div>
          <div><span>ADVANCING</span><strong className="mf-positive">{winners.length}<ArrowUpRight size={21}/></strong><p>{assets.length?Math.round(winners.length/assets.length*100):0}% of visible stocks</p></div>
          <div><span>DECLINING</span><strong className="mf-negative">{losers.length}<ArrowDownRight size={21}/></strong><p>{flat?`${flat} unchanged`:`${assets.length?Math.round(losers.length/assets.length*100):0}% of visible stocks`}</p></div>
          <div><span>AVERAGE SESSION MOVE</span><strong className={average>=0?"mf-positive":"mf-negative"}>{signed(average)}</strong><p>Equal-weight · from sample open</p></div>
          <div className="mf-session-stat"><span>SESSION PLAYHEAD</span><strong>{sessionTime(step)}<small>IST</small></strong><p>Sample session · 09:15–15:30</p></div>
        </div>
        <div className="mf-workbench">
          <div className="mf-stations" role="tablist" aria-label="Market views">{STATIONS.map(s=><button id={`mf-tab-${s.id}`} aria-controls="mf-view" role="tab" aria-selected={station===s.id} key={s.id} className={station===s.id?"is-active":""} onClick={()=>setStation(s.id)}><s.icon size={16}/><span>{s.short}</span>{s.id==="bullish"?<small className="mf-positive">{Math.min(20,winners.length)}</small>:s.id==="bearish"?<small className="mf-negative">{Math.min(20,losers.length)}</small>:null}</button>)}<button className="mf-stock-search" onClick={()=>{setQuery("");setSearch(true);}}><Search size={16}/><span>Find a stock</span><kbd>/</kbd></button></div>
          <div className="mf-view-toolbar"><div><span className="mf-status-ring"/><b>{station==="flow"?"SESSION FLOW":station==="market"?"MARKET PICTURE":station==="bullish"?"BULLISH CONTENDERS":"BEARISH CONTENDERS"}</b><span className="mf-toolbar-muted">{assets.length} stocks · {sessionTime(step)} IST</span></div><div><button className={`mf-label-toggle ${labels?"is-on":""}`} onClick={()=>setLabels(!labels)} aria-pressed={labels}><SlidersHorizontal size={13}/><span>Labels</span></button><button onClick={download} title="Download sample snapshot" aria-label="Download sample snapshot"><Download size={15}/></button><button onClick={()=>setExpanded(!expanded)} title={expanded?"Exit expanded view":"Expand view"} aria-label={expanded?"Exit expanded view":"Expand view"}><Expand size={15}/></button></div></div>
          {(sector||watchOnly)&&<div className="mf-active-filters">{sector&&<button onClick={()=>setSector(null)}>{sector}<X size={12}/></button>}{watchOnly&&<button onClick={()=>setWatchOnly(false)}>Watchlist<X size={12}/></button>}<span>All views reflect your filters.</span></div>}
          <div id="mf-view" role="tabpanel" aria-labelledby={`mf-tab-${station}`}>
            {!assets.length?<div className="mf-empty"><Star size={28}/><h2>No stocks in this view</h2><p>Add stocks to your watchlist from any stock detail, or clear the filters.</p><button onClick={()=>{setWatchOnly(false);setSector(null);}}>Show all stocks<ArrowRight size={14}/></button></div>:station==="flow"?<FlowCanvas assets={assets} step={step} selected={selected?.symbol??null} onSelect={inspect} showLabels={labels}/>:station==="market"?<div className="mf-market-picture"><div className="mf-market-caption"><div><span>MARKET BREADTH</span><h2>{winners.length>losers.length?"Buyers have the broader participation.":winners.length<losers.length?"Sellers have the broader participation.":"The market is balanced."}</h2><p>Each dot is one stock. Position shows its move from the sample open.</p></div><div className="mf-market-key"><span><i/>Advancing</span><span><i/>Declining</span></div></div><svg viewBox="0 0 1200 250" role="img" aria-label="Market breadth distribution"><line x1="80" x2="1120" y1="165" y2="165" stroke="#c7d2d5"/><line x1="600" x2="600" y1="30" y2="205" stroke="#c7d2d5"/>{[-4,-2,0,2,4].map(n=><text key={n} x={600+n*100} y="227" textAnchor="middle" className="mf-axis">{n>0?"+":""}{n}%</text>)}{assets.map((a,i)=>{const n=moveAt(a,step);return <g key={a.symbol} role="button" tabIndex={0} aria-label={`Inspect ${a.symbol}`} onClick={()=>inspect(a)} onKeyDown={e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();inspect(a);}}} className="mf-endpoint"><title>{a.symbol} {signed(n)}</title><circle cx={600+n*98} cy={n>=0?150-(i%7)*15:180+(i%3)*11} r={4+(i%3)} fill={n>=0?"#119b70":"#e25c77"} opacity=".7"/></g>;})}</svg></div>:<div className="mf-ranking-grid">{(station==="bullish"?winners:losers).slice(0,20).map((a,i)=><button key={a.symbol} className="mf-rank-card" onClick={()=>inspect(a)}><div className="mf-rank-top"><span className={station==="bullish"?"mf-rank-icon mf-up-bg":"mf-rank-icon mf-down-bg"}>{a.symbol.slice(0,2)}</span><span className="mf-rank-number">#{String(i+1).padStart(2,"0")}</span></div><b>{a.symbol}</b><span className="mf-rank-sector">{a.sector}</span><Sparkline asset={a} step={step}/><div className="mf-rank-bottom"><strong className={station==="bullish"?"mf-positive":"mf-negative"}>{signed(moveAt(a,step))}</strong><span>{rupees(priceAt(a,step))}</span></div></button>)}</div>}
          </div>
          <div className="mf-replay"><button className="mf-play" onClick={startReplay} aria-label={playing?"Pause session replay":"Replay sample session"}>{playing?<Pause size={14} fill="currentColor"/>:<Play size={14} fill="currentColor"/>}</button><span className="mf-replay-label">{playing?"REPLAYING":"SESSION REPLAY"}</span><span className="mf-replay-time">09:15</span><input aria-label="Session replay time" type="range" min="0" max={SESSION_STEPS} value={step} onChange={e=>{setPlaying(false);setStep(Number(e.target.value));}}/><b>{sessionTime(step)}</b><button onClick={()=>{setPlaying(false);setStep(SESSION_STEPS);}} aria-label="Reset to session close" title="Reset to session close"><RotateCcw size={14}/></button></div>
          <div className="mf-chart-footnote"><span><i className="mf-dot-up"/>Above open<i className="mf-dot-down"/>Below open</span><span>Paths show price performance, not transfers of money.<button onClick={()=>setInfo(true)} aria-label="Explain flow chart"><Info size={12}/></button></span></div>
        </div>
        <div className="mf-lower-grid"><section className="mf-sector-panel"><div className="mf-section-head"><div><Layers3 size={16}/><h2>Sector pulse</h2><span>{sectors.length}</span></div><button onClick={()=>setSector(null)} disabled={!sector}>View all<ArrowUpRight size={13}/></button></div><p className="mf-section-description">Where the session is gaining strength.</p><div className="mf-sector-grid">{sectors.map(s=><button key={s.name} className={`mf-sector-card ${sector===s.name?"is-selected":""}`} onClick={()=>setSector(sector===s.name?null:s.name)} aria-pressed={sector===s.name}><div><b>{s.name}</b><strong className={s.change>=0?"mf-positive":"mf-negative"}>{signed(s.change)}</strong></div><div className="mf-breadth-bars">{s.members.map(a=><i key={a.symbol} style={{background:moveAt(a,step)>0?"#36ae89":moveAt(a,step)<0?"#e68699":"#c8d2d7"}}/>)}</div><small>{s.up} of {s.members.length} advancing<ChevronRight size={11}/></small></button>)}</div></section>
        <section className="mf-leaders-panel"><div className="mf-section-head"><div><Activity size={16}/><h2>Session leaders</h2></div><span className="mf-small-sample">SAMPLE</span></div><p className="mf-section-description">The strongest moves in your current view.</p><div className="mf-leader-label"><TrendingUp size={13}/>LEADING THE MOVE</div>{winners.slice(0,3).map((a,i)=><button className="mf-leader-row" key={a.symbol} onClick={()=>inspect(a)}><span className="mf-leader-index">0{i+1}</span><div><b>{a.symbol}</b><small>{a.sector}</small></div><Sparkline asset={a} step={step}/><strong className="mf-positive">{signed(moveAt(a,step))}</strong><ChevronRight size={13}/></button>)}<div className="mf-leader-label mf-lagging"><TrendingDown size={13}/>LAGGING THE MARKET</div>{losers.slice(0,3).map((a,i)=><button className="mf-leader-row" key={a.symbol} onClick={()=>inspect(a)}><span className="mf-leader-index">0{i+1}</span><div><b>{a.symbol}</b><small>{a.sector}</small></div><Sparkline asset={a} step={step}/><strong className="mf-negative">{signed(moveAt(a,step))}</strong><ChevronRight size={13}/></button>)}{!ranked.length&&<p className="mf-section-description">No stocks match your filters.</p>}</section></div>
        <footer className="mf-footer"><span><span className="mf-footer-mark">S</span> SUPERTRADER RESEARCH</span><span>Designed for perspective. Built for inspection.</span><button onClick={()=>setInfo(true)}>About this preview<ArrowUpRight size={12}/></button></footer>
      </div>
    </main>
    {(selected||search||info)&&<div className="mf-modal-backdrop" onClick={()=>{setSelected(null);setSearch(false);setInfo(false);}}><div ref={dialogRef} className={`mf-dialog ${search?"mf-search-dialog":""}`} role="dialog" aria-modal="true" aria-label={selected?`${selected.symbol} stock detail`:search?"Find a stock":"Data and methodology"} onClick={e=>e.stopPropagation()}>
      <button ref={closeDetail} className="mf-dialog-close" aria-label="Close dialog" onClick={()=>{setSelected(null);setSearch(false);setInfo(false);}}><X size={20}/></button>
      {selected?<><div className="mf-detail-heading"><span className={`mf-detail-symbol ${detailPositive?"mf-up-bg":"mf-down-bg"}`}>{selected.symbol.slice(0,2)}</span><div><span className="mf-eyebrow">{selected.sector} · SAMPLE</span><h2>{selected.symbol}</h2><p>{selected.name}</p></div></div><div className="mf-detail-price"><strong>{rupees(priceAt(selected,step))}</strong><span className={detailPositive?"mf-positive":"mf-negative"}>{signed(moveAt(selected,step))}<small>from sample open</small></span></div><div className="mf-detail-chart"><Sparkline asset={selected} step={step} large/><div><span>09:15</span><span>{sessionTime(step)} IST</span></div></div><div className="mf-detail-metrics"><div><span>SESSION OPEN</span><b>{rupees(selected.open)}</b></div><div><span>HIGHEST SAMPLE CLOSE</span><b>{rupees(selected.open*(1+Math.max(0,...selected.series.slice(0,step+1))/100))}</b></div><div><span>LOWEST SAMPLE CLOSE</span><b>{rupees(selected.open*(1+Math.min(0,...selected.series.slice(0,step+1))/100))}</b></div></div><div className="mf-detail-notice"><Info size={15}/><p>Illustrative price path. Order-block zones, volume and open interest require a connected, validated market feed.</p></div><div className="mf-detail-actions"><button className="mf-primary" onClick={()=>toggleWatch(selected.symbol)}>{watchlist.includes(selected.symbol)?<Check size={15}/>:<Star size={15}/>} {watchlist.includes(selected.symbol)?"Saved to watchlist":"Add to watchlist"}</button><button onClick={()=>{setSector(selected.sector);setSelected(null);}}>Explore {selected.sector}<ArrowRight size={14}/></button></div></>:search?<><div className="mf-search-input"><Search size={20}/><input ref={searchInput} placeholder="Search stocks, names or sectors…" aria-label="Search stocks" value={query} onChange={e=>setQuery(e.target.value)}/></div><div className="mf-search-meta">{searchResults.length} stocks · illustrative data</div><div className="mf-search-results">{searchResults.map(a=><button key={a.symbol} onClick={()=>inspect(a)}><span className="mf-search-monogram">{a.symbol.slice(0,2)}</span><div><b>{a.symbol}</b><small>{a.name} · {a.sector}</small></div><strong className={moveAt(a,step)>=0?"mf-positive":"mf-negative"}>{signed(moveAt(a,step))}</strong><ChevronRight size={14}/></button>)}{!searchResults.length&&<div className="mf-empty"><Search size={24}/><h2>No matching stocks</h2><p>Try a symbol such as SBIN or a sector such as Banking.</p></div>}</div></>:<><span className="mf-eyebrow">DATA & METHODOLOGY</span><h2>A map of price movement.</h2><p>This first version is an interactive design preview. Every price and session trajectory is synthetic. Stock names provide a familiar sample universe; they are not current quotes.</p><div className="mf-method-item"><span>01</span><div><h3>Read the paths</h3><p>Each path is one stock’s percentage change from the sample session open: (price ÷ open − 1) × 100. Green endpoints finish above open; red endpoints finish below it.</p></div></div><div className="mf-method-item"><span>02</span><div><h3>Follow participation</h3><p>Sector returns are equal-weight averages of the included stocks. Each small bar represents a constituent. Sector classifications and this curated universe are illustrative.</p></div></div><div className="mf-method-item"><span>03</span><div><h3>Replay with context</h3><p>The playhead updates paths, breadth, rankings, sector returns and detail prices to the selected time. Boards rank by percentage move; they do not claim institutional activity or an order-block signal.</p></div></div><div className="mf-method-item"><span>04</span><div><h3>Connect the next layer</h3><p>The production stage needs a licensed market feed, exchange session calendars, completed-bar history, verified symbol mapping and freshness checks. No live feed is connected in this preview.</p></div></div><button className="mf-primary" onClick={()=>setInfo(false)}>Got it<Check size={15}/></button></>}
    </div></div>}
  </div>;
}
