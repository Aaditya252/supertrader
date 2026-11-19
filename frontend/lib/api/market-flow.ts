import type { FlowAsset } from "@/lib/market-flow";
import { DEMO_ASSETS } from "@/data/market-flow-preview";

export interface MarketFlowSnapshot {
  mode: "synthetic_preview";
  source: "local_design_fixture";
  timezone: "Asia/Kolkata";
  sessionOpen: "09:15";
  sessionClose: "15:30";
  intervalMinutes: 5;
  assets: FlowAsset[];
}

/** Replace this adapter when the production backend is connected. */
export function getMarketFlowSnapshot(): MarketFlowSnapshot {
  return {
    mode: "synthetic_preview", source: "local_design_fixture", timezone: "Asia/Kolkata",
    sessionOpen: "09:15", sessionClose: "15:30", intervalMinutes: 5, assets: DEMO_ASSETS,
  };
}
