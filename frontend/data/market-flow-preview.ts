import type { FlowAsset } from "@/lib/market-flow";
const SESSION_STEPS = 75;
const universe: [string, string, string, number, number][] = [
  ["HDFCBANK","HDFC Bank","Banking",1680,2.86],["ICICIBANK","ICICI Bank","Banking",1240,2.19],["SBIN","State Bank of India","Banking",810,3.24],["AXISBANK","Axis Bank","Banking",1120,1.78],["KOTAKBANK","Kotak Mahindra Bank","Banking",1840,.82],["INDUSINDBK","IndusInd Bank","Banking",1040,-.64],
  ["TATASTEEL","Tata Steel","Metals",158,4.82],["HINDALCO","Hindalco Industries","Metals",654,3.16],["JSWSTEEL","JSW Steel","Metals",945,2.72],["JINDALSTEL","Jindal Steel","Metals",910,1.98],["SAIL","Steel Authority of India","Metals",124,1.21],["VEDL","Vedanta","Metals",448,-.32],
  ["TCS","Tata Consultancy Services","Technology",3890,-2.18],["INFY","Infosys","Technology",1820,-3.12],["HCLTECH","HCL Technologies","Technology",1740,-1.34],["WIPRO","Wipro","Technology",286,-2.74],["TECHM","Tech Mahindra","Technology",1610,.41],["LTIM","LTIMindtree","Technology",5890,-1.89],
  ["MARUTI","Maruti Suzuki","Automobiles",12450,1.36],["M&M","Mahindra & Mahindra","Automobiles",2890,2.51],["TATAMOTORS","Tata Motors","Automobiles",780,.94],["BAJAJ-AUTO","Bajaj Auto","Automobiles",9250,1.72],["EICHERMOT","Eicher Motors","Automobiles",4890,-.72],["HEROMOTOCO","Hero MotoCorp","Automobiles",4580,.48],
  ["RELIANCE","Reliance Industries","Energy",1280,1.68],["ONGC","Oil & Natural Gas Corp","Energy",264,2.12],["BPCL","Bharat Petroleum","Energy",304,.74],["IOC","Indian Oil","Energy",142,-.26],["GAIL","GAIL India","Energy",198,1.19],["OIL","Oil India","Energy",468,2.46],
  ["SUNPHARMA","Sun Pharmaceutical","Healthcare",1780,-.62],["DRREDDY","Dr. Reddy’s Laboratories","Healthcare",1240,-1.18],["CIPLA","Cipla","Healthcare",1480,.87],["DIVISLAB","Divi’s Laboratories","Healthcare",5890,-1.52],["LUPIN","Lupin","Healthcare",2190,-.34],["AUROPHARMA","Aurobindo Pharma","Healthcare",1190,.28],
  ["HINDUNILVR","Hindustan Unilever","Consumer",2480,-1.32],["ITC","ITC","Consumer",420,.32],["NESTLEIND","Nestlé India","Consumer",2290,-.89],["BRITANNIA","Britannia Industries","Consumer",5480,-1.76],["TATACONSUM","Tata Consumer Products","Consumer",1040,-.47],["DABUR","Dabur India","Consumer",524,-2.21],
  ["NTPC","NTPC","Power",348,2.41],["POWERGRID","Power Grid Corp","Power",294,1.92],["TATAPOWER","Tata Power","Power",410,1.37],["ADANIPOWER","Adani Power","Power",540,3.67],["NHPC","NHPC","Power",82,.65],["SJVN","SJVN","Power",108,-.54],
  ["DLF","DLF","Real Estate",845,-2.84],["GODREJPROP","Godrej Properties","Real Estate",2780,-1.91],["OBEROIRLTY","Oberoi Realty","Real Estate",1940,-1.29],["PRESTIGE","Prestige Estates","Real Estate",1640,-3.96],["PHOENIXLTD","Phoenix Mills","Real Estate",1680,.39],["LODHA","Lodha Developers","Real Estate",1240,-2.47],
  ["BAJFINANCE","Bajaj Finance","Financials",920,1.12],["BAJAJFINSV","Bajaj Finserv","Financials",1980,.71],["CHOLAFIN","Cholamandalam Finance","Financials",1390,1.48],["SHRIRAMFIN","Shriram Finance","Financials",610,-.43],["PFC","Power Finance Corp","Financials",468,2.27],["RECLTD","REC","Financials",510,1.82],
  ["BHARTIARTL","Bharti Airtel","Telecom",1680,2.56],["IDEA","Vodafone Idea","Telecom",9.4,-1.48],["INDUSTOWER","Indus Towers","Telecom",368,1.94],
  ["HAL","Hindustan Aeronautics","Industrials",4380,-2.61],["BEL","Bharat Electronics","Industrials",294,.92],["LT","Larsen & Toubro","Industrials",3640,-.85],["BHEL","Bharat Heavy Electricals","Industrials",218,-3.41],["SIEMENS","Siemens","Industrials",6480,-1.27],["ABB","ABB India","Industrials",6890,-.23],
];

export const DEMO_ASSETS: FlowAsset[] = universe.map(([symbol,name,sector,open,end],i) => {
  const series = Array.from({length:SESSION_STEPS+1},(_,step) => {
    const t=step/SESSION_STEPS;
    const delay=.2+(i%9)*.065,steepness=8+(i%5)*2;
    const curve=(v:number)=>1/(1+Math.exp(-steepness*(v-delay)));
    const progress=(curve(t)-curve(0))/(curve(1)-curve(0));
    const wave=(Math.sin(t*(8+i%5)+i*.8)-Math.sin(i*.8))*.3;
    const pulse=Math.sin(t*27+i)*Math.sin(Math.PI*t)*.11;
    return end*progress + wave*Math.sin(Math.PI*t) + pulse;
  });
  return {symbol,name,sector,open,series};
});

