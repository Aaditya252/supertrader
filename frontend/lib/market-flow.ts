/** Design fixtures only. These are not exchange observations. */
export type FlowAsset = { symbol: string; name: string; sector: string; open: number; series: number[] };
export type FlowStation = "market" | "flow" | "bullish" | "bearish";
export const SESSION_LABELS = ["09:15", "10:00", "11:00", "12:00", "13:00", "14:00", "15:30"];
export const SESSION_STEPS = 75;

export const moveAt=(asset:FlowAsset,step:number)=>asset.series[Math.max(0,Math.min(SESSION_STEPS,step))];
export const priceAt=(asset:FlowAsset,step:number)=>asset.open*(1+moveAt(asset,step)/100);
export const signed=(n:number)=>`${n>=0?"+":"−"}${Math.abs(n).toFixed(2)}%`;
export const sessionTime=(step:number)=>{
  const mins=9*60+15+step*5;
  return `${Math.floor(mins/60).toString().padStart(2,"0")}:${(mins%60).toString().padStart(2,"0")}`;
};
export function sectorStats(assets:FlowAsset[],step:number){
  const groups=new Map<string,FlowAsset[]>();
  for(const a of assets) groups.set(a.sector,[...(groups.get(a.sector)??[]),a]);
  return Array.from(groups,([name,members])=>({name,members,change:members.reduce((sum,a)=>sum+moveAt(a,step),0)/members.length,up:members.filter(a=>moveAt(a,step)>0).length})).sort((a,b)=>b.change-a.change);
}
