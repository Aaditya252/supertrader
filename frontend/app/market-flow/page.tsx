import type { Metadata } from "next";
import MarketFlow from "@/components/market-flow/MarketFlow";
import "./market-flow.css";

export const metadata: Metadata = {title:"Market Flow · Supertrader",description:"Explore market breadth, sector leadership and intraday price trajectories."};
export default function MarketFlowPage(){return <MarketFlow/>;}
