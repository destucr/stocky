import { useState, useEffect, useRef } from 'react';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Container, 
  Paper, 
  Box, 
  Select, 
  MenuItem, 
  FormControl, 
  InputLabel,
  Chip,
  ToggleButtonGroup,
  ToggleButton,
  Tooltip,
  IconButton,
  Button
} from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import { createChart, ColorType, IChartApi, ISeriesApi, Time, PriceScaleMode, MouseEventParams } from 'lightweight-charts';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import BarChartIcon from '@mui/icons-material/BarChart';
import TimelineIcon from '@mui/icons-material/Timeline';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import theme, { UI_COLORS as COLORS } from './theme';
import StockLegend from './components/StockLegend';

// Types corresponding to backend
interface TradeData {
  p: number; // Price
  s: string; // Symbol
  t: number; // Time
  v: number; // Volume
}

interface BackendMessage {
  type: string;
  data: TradeData[];
  msg?: string;
  fetched_at?: number; // Added for latency tracking
}

interface SymbolMetadata {
  name: string;
  logo: string;
}

// Candle structure for the chart
interface Candle {
  time: Time;
  open: number;
  high: number;
  low: number;
  close: number;
}

const calculateMA = (data: any[], period: number) => {
    const maData = [];
    for (let i = 0; i < data.length; i++) {
      if (i < period - 1) continue;
      let sum = 0;
      for (let j = 0; j < period; j++) {
        sum += data[i - j].close;
      }
      maData.push({ time: data[i].time, value: sum / period });
    }
    return maData;
};

const AVAILABLE_TIMEFRAMES = [
  { label: '1 Second', value: 1 },
  { label: '5 Seconds', value: 5 },
  { label: '10 Seconds', value: 10 },
  { label: '1 Minute', value: 60 },
  { label: '5 Minutes', value: 300 },
  { label: '1 Hour', value: 3600 },
  { label: '1 Day', value: 86400 },
];

const UI_COLORS = {
  background: '#0B0E11',
  surface: '#181A20',
  surfaceLight: '#1E2329',
  border: '#2B2F36',
  borderLight: '#474D57',
  textPrimary: '#EAECEF',
  textSecondary: '#848E9C',
  textDisabled: '#474D57',
  accent: '#F0B90B',
  success: '#26a69a',
  danger: '#ef5350',
  ma7: '#F0B90B',
  ma25: '#9261F2',
  ma99: '#2962FF',
  overlayBg: 'rgba(24, 26, 32, 0.85)',
  overlayBorder: 'rgba(71, 77, 87, 0.3)',
};

function App() {
  const [activeSymbol, setActiveSymbol] = useState<string>('');
  const [availableSymbols, setAvailableSymbols] = useState<string[]>([]);
  const [symbolMetadata, setSymbolMetadata] = useState<Record<string, SymbolMetadata>>({});
  const symbolMetadataRef = useRef<Record<string, SymbolMetadata>>({});
  const [isConnected, setIsConnected] = useState(false);
  const [chartType, setChartType] = useState<'candlestick' | 'line'>('candlestick');
  const [isLogScale, setIsLogScale] = useState(false);
  const [timeframe, setTimeframe] = useState<number>(60);
  const timeframeRef = useRef(timeframe);
  const lastSymbolRef = useRef<string>('');
  const [isAutoScale, setIsAutoScale] = useState(true);
  const lastPriceRef = useRef<number | null>(null);
  const isAutoScaleRef = useRef(true);
  const lastRangeRef = useRef<{ from: number, to: number } | null>(null);
  
  // State for legend - replacement for innerHTML
  const [legendData, setLegendData] = useState<{
    candle: any;
    mas: any;
  }>({ candle: null, mas: {} });

  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const legendRef = useRef<HTMLDivElement>(null); // Ref for the legend
  
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const lineSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const ma7SeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const ma25SeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const ma99SeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  
  // State to hold current building candle
  const currentCandleRef = useRef<(Candle & { volume: number }) | null>(null);
  const lastCandleTimeRef = useRef<number>(0);
  
  // For calculating MAs live
  const candleHistoryRef = useRef<any[]>([]);
  const ma7HistoryRef = useRef<any[]>([]);
  const ma25HistoryRef = useRef<any[]>([]);
  const ma99HistoryRef = useRef<any[]>([]);

  // Sync timeframeRef
  useEffect(() => {
    timeframeRef.current = timeframe;
  }, [timeframe]);

  // Fetch metadata on mount
  useEffect(() => {
    const fetchMetadata = async () => {
      try {
        console.log('Fetching symbol metadata...');
        const response = await fetch('http://localhost:8080/api/metadata');
        if (response.ok) {
          const data = await response.json();
          console.log('Metadata received:', data);
          setSymbolMetadata(data);
          symbolMetadataRef.current = data;
        } else {
          console.error('Metadata API error:', response.status);
        }
      } catch (err) {
        console.error('Failed to fetch metadata:', err);
      }
    };
    fetchMetadata();
  }, []);

  // Update ref whenever state changes to ensure closures see latest data
  useEffect(() => {
    symbolMetadataRef.current = symbolMetadata;
    // If we already have a live candle, update the legend immediately to show the new logo/name
    if (currentCandleRef.current && activeSymbol) {
        updateLegendUI(
            currentCandleRef.current, 
            activeSymbol,
            ma7HistoryRef.current[ma7HistoryRef.current.length - 1]?.value,
            ma25HistoryRef.current[ma25HistoryRef.current.length - 1]?.value,
            ma99HistoryRef.current[ma99HistoryRef.current.length - 1]?.value
        );
    }
  }, [symbolMetadata, activeSymbol]);

  // Legend Updater - Now uses React state
  const updateLegendUI = (candle: any, _symbol: string, ma7?: number, ma25?: number, ma99?: number) => {
    setLegendData({
        candle,
        mas: { ma7, ma25, ma99 }
    });
  };

  // Initialize Chart
  useEffect(() => {
    if (chartContainerRef.current) {
      const chart = createChart(chartContainerRef.current, {
        layout: {
          background: { type: ColorType.Solid, color: UI_COLORS.background },
          textColor: UI_COLORS.textSecondary,
        },
        grid: {
          vertLines: { color: UI_COLORS.border },
          horzLines: { color: UI_COLORS.border },
        },
        width: chartContainerRef.current.clientWidth,
        height: chartContainerRef.current.clientHeight || 500,
        timeScale: {
          timeVisible: true,
          secondsVisible: true,
          borderColor: UI_COLORS.borderLight,
          shiftVisibleRangeOnNewBar: true, // Auto-scroll to new candles
          rightOffset: 10, // Leave space on the right
          barSpacing: 10,  // Comfortable default spacing
        },
        rightPriceScale: {
          borderColor: UI_COLORS.borderLight,
          autoScale: true,
          alignLabels: true,
        },
        handleScale: {
          mouseWheel: false, // Disable default horizontal zoom via wheel
          pinch: true,
          axisPressedMouseMove: true,
        },
        handleScroll: {
          mouseWheel: false, // Disable default scroll
          pressedMouseMove: true,
        },
      });

      const candlestickSeries = chart.addCandlestickSeries({
        upColor: COLORS.success, 
        downColor: COLORS.danger, 
        borderVisible: false, 
        wickUpColor: COLORS.success, 
        wickDownColor: COLORS.danger, 
        visible: chartType === 'candlestick',
        priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
      });

      const lineSeries = chart.addLineSeries({
        color: COLORS.accent,
        lineWidth: 2,
        visible: chartType === 'line',
      });

      const volumeSeries = chart.addHistogramSeries({
        color: COLORS.success,
        priceFormat: { type: 'volume' },
        priceScaleId: '', 
      });

      const ma7Series = chart.addLineSeries({
        color: COLORS.ma7,
        lineWidth: 1,
        title: 'MA7',
        visible: true,
      });

      const ma25Series = chart.addLineSeries({
        color: COLORS.ma25,
        lineWidth: 1,
        title: 'MA25',
        visible: true,
      });

      const ma99Series = chart.addLineSeries({
        color: COLORS.ma99,
        lineWidth: 1,
        title: 'MA99',
        visible: true,
      });

      volumeSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      });

      chartRef.current = chart;
      seriesRef.current = candlestickSeries;
      lineSeriesRef.current = lineSeries;
      volumeSeriesRef.current = volumeSeries;
      ma7SeriesRef.current = ma7Series;
      ma25SeriesRef.current = ma25Series;
      ma99SeriesRef.current = ma99Series;

      const handleCrosshair = (param: MouseEventParams) => {
        if (!legendRef.current) return;
        
        // Track last price for centered zoom
        if (param.point && seriesRef.current) {
          lastPriceRef.current = seriesRef.current.coordinateToPrice(param.point.y);
        }

        const validPoint = param.time && param.point && param.point.x >= 0;

        if (validPoint) {
            const candleData: any = param.seriesData.get(candlestickSeries);
            const lineData: any = param.seriesData.get(lineSeries);
            const volumeData: any = param.seriesData.get(volumeSeries);
            const ma7Data: any = param.seriesData.get(ma7Series);
            const ma25Data: any = param.seriesData.get(ma25Series);
            const ma99Data: any = param.seriesData.get(ma99Series);
            
            if (candleData || lineData) {
                const data = candleData || { open: lineData.value, high: lineData.value, low: lineData.value, close: lineData.value };
                updateLegendUI(
                    { ...data, volume: volumeData?.value }, 
                    activeSymbolRef.current,
                    ma7Data ? ma7Data.value : undefined,
                    ma25Data ? ma25Data.value : undefined,
                    ma99Data ? ma99Data.value : undefined
                );
                return;
            }
        }
        
        // Fallback to current live candle
        if (currentCandleRef.current) {
            updateLegendUI(
                currentCandleRef.current, 
                activeSymbolRef.current,
                ma7HistoryRef.current[ma7HistoryRef.current.length - 1]?.value,
                ma25HistoryRef.current[ma25HistoryRef.current.length - 1]?.value,
                ma99HistoryRef.current[ma99HistoryRef.current.length - 1]?.value
            );
        }
      };

      chart.subscribeCrosshairMove(handleCrosshair);

      // --- Real-time Vertical Price Zoom Logic ---
      const handleWheel = (e: WheelEvent) => {
        if (!chartRef.current || !seriesRef.current || !container) return;
        
        // Take over wheel for vertical zoom as requested
        e.preventDefault();

        const priceScale = chartRef.current.priceScale('right');
        
        // In autoScale mode, options().priceRange is null. 
        // We use our tracked lastRangeRef as the starting point for scaling.
        // @ts-ignore
        const currentOptions = priceScale.options();
        // @ts-ignore
        const range = (currentOptions && currentOptions.priceRange) || lastRangeRef.current;
        
        if (!range) return;

        const rect = container.getBoundingClientRect();
        const mouseY = e.clientY - rect.top;
        
        // Anchored price at the moment of the scroll event
        const focusPrice = seriesRef.current.coordinateToPrice(mouseY);
        if (focusPrice === null) return;

        // Map-like exponential zoom factor
        const delta = e.deltaY * (e.deltaMode === 1 ? 20 : 1);
        const factor = Math.pow(1.0015, delta);
        
        const currentRange = range.to - range.from;
        const newRangeSize = currentRange * factor;

        // Clamp to prevent extreme scaling
        if (newRangeSize < 0.001 || newRangeSize > 10000000) return;

        // Calculate new range while keeping focusPrice at same relative screen position
        const relativePos = (focusPrice - range.from) / currentRange;
        const newFrom = focusPrice - relativePos * newRangeSize;
        const newTo = focusPrice + (1 - relativePos) * newRangeSize;

        const newRange = { from: newFrom, to: newTo };
        lastRangeRef.current = newRange;

        // Switch to manual mode and defer React update to keep event loop fast
        if (isAutoScaleRef.current) {
          isAutoScaleRef.current = false;
          setTimeout(() => setIsAutoScale(false), 0);
        }

        // Apply visual stretching immediately via applyOptions
        priceScale.applyOptions({ 
            autoScale: false,
            // @ts-ignore
            priceRange: newRange 
        });
      };

      const container = chartContainerRef.current;
      if (container) {
        container.addEventListener('wheel', handleWheel, { passive: false });
      }

      const handleResize = () => {
        if (chartContainerRef.current && chartRef.current) {
          chartRef.current.applyOptions({ 
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight 
          });
        }
      };

      window.addEventListener('resize', handleResize);
      return () => {
        window.removeEventListener('resize', handleResize);
        if (container) {
          container.removeEventListener('wheel', handleWheel);
        }
        chart.remove();
      };
    }
  }, []);

  // Handle Chart Type Change
  useEffect(() => {
    if (seriesRef.current && lineSeriesRef.current) {
      seriesRef.current.applyOptions({ visible: chartType === 'candlestick' });
      lineSeriesRef.current.applyOptions({ visible: chartType === 'line' });
    }
  }, [chartType]);

  // Handle Scale Change
  useEffect(() => {
    if (chartRef.current) {
      chartRef.current.priceScale('right').applyOptions({
        mode: isLogScale ? PriceScaleMode.Logarithmic : PriceScaleMode.Normal,
      });
    }
  }, [isLogScale]);

  // WebSocket Connection
  const activeSymbolRef = useRef(activeSymbol);
  useEffect(() => {
    activeSymbolRef.current = activeSymbol;
  }, [activeSymbol]);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimeout: number | null = null;
    let isComponentMounted = true;

    const connect = () => {
      if (!isComponentMounted) return;
      
      ws = new WebSocket('ws://localhost:8080/ws');
      ws.onopen = () => {
        if (isComponentMounted) setIsConnected(true);
      };
      ws.onclose = () => {
        if (isComponentMounted) {
          setIsConnected(false);
          reconnectTimeout = window.setTimeout(connect, 3000);
        }
      };
      ws.onmessage = (event) => {
        if (!isComponentMounted) return;
        try {
          const message: BackendMessage = JSON.parse(event.data);
          if (message.type === 'trade') {
            const receivedSymbols = new Set(message.data.map(t => t.s));
            setAvailableSymbols(prev => {
              const newSymbols = Array.from(receivedSymbols).filter(s => !prev.includes(s));
              return newSymbols.length > 0 ? [...prev, ...newSymbols].sort() : prev;
            });

            message.data.forEach(trade => {
              if (activeSymbolRef.current && trade.s === activeSymbolRef.current) {
                processTrade(trade, message.fetched_at);
              }
            });
          }
        } catch (err) { console.error("WS Error:", err); }
      };
    };

    connect();
    return () => {
      isComponentMounted = false;
      if (ws) {
        ws.onclose = null; // Prevent reconnection on cleanup
        ws.close();
      }
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
    };
  }, []);

  // If no symbol selected, select the first one found
  useEffect(() => {
    if (!activeSymbol && availableSymbols.length > 0) {
      setActiveSymbol(availableSymbols[0]);
    }
  }, [availableSymbols, activeSymbol]);

  useEffect(() => {
    // Reset building state when symbol or timeframe changes
    currentCandleRef.current = null;
    lastCandleTimeRef.current = 0;
    candleHistoryRef.current = [];
    ma7HistoryRef.current = [];
    ma25HistoryRef.current = [];
    ma99HistoryRef.current = [];

    // Restore auto-scale on changes
    if (chartRef.current) {
        chartRef.current.priceScale('right').applyOptions({ autoScale: true });
        setIsAutoScale(true);
        isAutoScaleRef.current = true;
    }

    // If symbol changed, we MUST clear everything immediately to avoid mixing symbols
    if (activeSymbol && activeSymbol !== lastSymbolRef.current) {
      if (seriesRef.current) {
        seriesRef.current.setData([]);
        lineSeriesRef.current?.setData([]);
        volumeSeriesRef.current?.setData([]);
        ma7SeriesRef.current?.setData([]);
        ma25SeriesRef.current?.setData([]);
        ma99SeriesRef.current?.setData([]);
      }
      lastSymbolRef.current = activeSymbol;
    }

    // Always fetch new history for the selected timeframe
    if (activeSymbol) {
      fetchHistory(activeSymbol, timeframe);
    }
  }, [activeSymbol, timeframe]);

  const handleResetScale = () => {
    if (chartRef.current) {
      const priceScale = chartRef.current.priceScale('right');
      priceScale.applyOptions({ 
        autoScale: true,
        // @ts-ignore
        priceRange: null 
      });
      lastRangeRef.current = null;
      setIsAutoScale(true);
      isAutoScaleRef.current = true;
    }
  };

  const fetchHistory = async (symbol: string, interval: number) => {
    try {
      // Request more candles for smaller timeframes to keep the chart full
      let limit = 500;
      if (interval <= 1) limit = 2000;
      else if (interval <= 10) limit = 1000;
      else if (interval <= 60) limit = 500;

      const url = `http://localhost:8080/api/history?symbol=${encodeURIComponent(symbol)}&interval=${interval}&limit=${limit}`;
      const response = await fetch(url);
      if (!response.ok) throw new Error('Failed to fetch history');
      const data = await response.json();
      
      if (Array.isArray(data) && data.length > 0) {
        const candles: any[] = [];
        const lineData: any[] = [];
        const volumeData: any[] = [];

        data.forEach((c: any) => {
          const t = Math.floor(new Date(c.time).getTime() / 1000) as Time;
          candles.push({ time: t, open: c.open, high: c.high, low: c.low, close: c.close });
          lineData.push({ time: t, value: c.close });
          volumeData.push({ time: t, value: c.volume, color: c.close >= c.open ? '#26a69a' : '#ef5350' });
        });

        candleHistoryRef.current = candles;
        const ma7 = calculateMA(candles, 7);
        const ma25 = calculateMA(candles, 25);
        const ma99 = calculateMA(candles, 99);

        ma7HistoryRef.current = ma7;
        ma25HistoryRef.current = ma25;
        ma99HistoryRef.current = ma99;

        // Update series with new resolution
        seriesRef.current?.setData(candles);
        lineSeriesRef.current?.setData(lineData);
        volumeSeriesRef.current?.setData(volumeData);
        ma7SeriesRef.current?.setData(ma7);
        ma25SeriesRef.current?.setData(ma25);
        ma99SeriesRef.current?.setData(ma99);

        const last = data[data.length - 1];
        lastCandleTimeRef.current = Math.floor(new Date(last.time).getTime() / 1000);
        currentCandleRef.current = {
          time: lastCandleTimeRef.current as Time,
          open: last.open, high: last.high, low: last.low, close: last.close,
          volume: last.volume
        };
        updateLegendUI(
            currentCandleRef.current, 
            symbol,
            ma7.length > 0 ? ma7[ma7.length - 1].value : undefined,
            ma25.length > 0 ? ma25[ma25.length - 1].value : undefined,
            ma99.length > 0 ? ma99[ma99.length - 1].value : undefined
        );

        // Snap to latest data
        chartRef.current?.timeScale().scrollToRealTime();

        // Initialize lastRangeRef for zoom anchoring by calculating min/max from data
        if (candles.length > 0) {
            let min = candles[0].low;
            let max = candles[0].high;
            for (let i = 1; i < candles.length; i++) {
                if (candles[i].low < min) min = candles[i].low;
                if (candles[i].high > max) max = candles[i].high;
            }
            // Add a small margin
            const padding = (max - min) * 0.05;
            lastRangeRef.current = { from: min - padding, to: max + padding };
        }
      } else {
        // Clear if no data
        seriesRef.current?.setData([]);
        lineSeriesRef.current?.setData([]);
        volumeSeriesRef.current?.setData([]);
        ma7SeriesRef.current?.setData([]);
        ma25SeriesRef.current?.setData([]);
        ma99SeriesRef.current?.setData([]);
        lastRangeRef.current = null;
      }
    } catch (err) { console.error("History Error:", err); }
  };

  const processTrade = (trade: TradeData, fetchedAt?: number) => {
    const tradeTime = Math.floor(trade.t / 1000);
    const candleInterval = timeframeRef.current; 
    const candleTime = Math.floor(tradeTime / candleInterval) * candleInterval;

    const updateMAs = (currentClose: number, time: Time, isNewCandle: boolean) => {
        // We use a copy of candle history + the current active candle to calculate MAs
        const history = [...candleHistoryRef.current];
        if (isNewCandle) {
            // If it's a new candle, the previous one is now "history"
            // We'll update the history ref when a new candle *starts*
        }
        
        // For the purpose of calculation, we append the "live" candle to history
        const virtualHistory = [...history, { time, close: currentClose }];
        
        const calcLastMA = (period: number) => {
            if (virtualHistory.length < period) return null;
            let sum = 0;
            for (let i = 0; i < period; i++) {
                sum += virtualHistory[virtualHistory.length - 1 - i].close;
            }
            return sum / period;
        };

        const v7 = calcLastMA(7);
        const v25 = calcLastMA(25);
        const v99 = calcLastMA(99);

        if (v7 !== null) {
            ma7SeriesRef.current?.update({ time, value: v7 });
            if (isNewCandle) ma7HistoryRef.current.push({ time, value: v7 });
        }
        if (v25 !== null) {
            ma25SeriesRef.current?.update({ time, value: v25 });
            if (isNewCandle) ma25HistoryRef.current.push({ time, value: v25 });
        }
        if (v99 !== null) {
            ma99SeriesRef.current?.update({ time, value: v99 });
            if (isNewCandle) ma99HistoryRef.current.push({ time, value: v99 });
        }

        return { v7, v25, v99 };
    };

    let mas = { v7: null as any, v25: null as any, v99: null as any };

    if (!currentCandleRef.current || candleTime > lastCandleTimeRef.current) {
      // Finalize the previous candle if it existed
      if (currentCandleRef.current) {
        candleHistoryRef.current.push({ 
            time: currentCandleRef.current.time, 
            close: currentCandleRef.current.close 
        });
      }

      const newCandle = {
        time: candleTime as Time,
        open: trade.p, high: trade.p, low: trade.p, close: trade.p,
        volume: trade.v,
      };
      currentCandleRef.current = newCandle;
      lastCandleTimeRef.current = candleTime;
      seriesRef.current?.update(newCandle);
      lineSeriesRef.current?.update({ time: candleTime as Time, value: trade.p });
      volumeSeriesRef.current?.update({ time: candleTime as Time, value: trade.v, color: '#26a69a' });
      
      mas = updateMAs(trade.p, candleTime as Time, true);

    } else if (candleTime === lastCandleTimeRef.current) {
      const c = currentCandleRef.current;
      c.high = Math.max(c.high, trade.p);
      c.low = Math.min(c.low, trade.p);
      c.close = trade.p;
      c.volume += trade.v;
      c.time = lastCandleTimeRef.current as Time;
      
      seriesRef.current?.update(c);
      lineSeriesRef.current?.update({ time: c.time, value: trade.p });
      volumeSeriesRef.current?.update({
        time: c.time, value: c.volume,
        color: c.close >= c.open ? '#26a69a' : '#ef5350'
      });

      mas = updateMAs(trade.p, c.time, false);
    }
    updateLegendUI(currentCandleRef.current, activeSymbolRef.current, mas.v7, mas.v25, mas.v99);

    if (fetchedAt) {
        const latency = Date.now() - fetchedAt;
        // Optional: log latency to console for monitoring (throttled to avoid spam)
        if (Math.random() < 0.05) { // 5% sample rate
            console.log(`[Latency] ${latency}ms from backend fetch to chart draw`);
        }
    }
  };

  const handleZoomIn = () => {
    const currentIndex = AVAILABLE_TIMEFRAMES.findIndex(tf => tf.value === timeframe);
    if (currentIndex > 0) {
      setTimeframe(AVAILABLE_TIMEFRAMES[currentIndex - 1].value);
    } else if (chartRef.current) {
      const timeScale = chartRef.current.timeScale();
      const logicalRange = timeScale.getVisibleLogicalRange();
      if (logicalRange) {
        const span = logicalRange.to - logicalRange.from;
        const center = (logicalRange.to + logicalRange.from) / 2;
        const newSpan = span * 0.8; 
        timeScale.setVisibleLogicalRange({
          from: center - newSpan / 2,
          to: center + newSpan / 2,
        });
      }
    }
  };

  const handleZoomOut = () => {
    const currentIndex = AVAILABLE_TIMEFRAMES.findIndex(tf => tf.value === timeframe);
    if (currentIndex < AVAILABLE_TIMEFRAMES.length - 1) {
      setTimeframe(AVAILABLE_TIMEFRAMES[currentIndex + 1].value);
    } else if (chartRef.current) {
      const timeScale = chartRef.current.timeScale();
      const logicalRange = timeScale.getVisibleLogicalRange();
      if (logicalRange) {
        const span = logicalRange.to - logicalRange.from;
        const center = (logicalRange.to + logicalRange.from) / 2;
        const newSpan = span * 1.25; 
        timeScale.setVisibleLogicalRange({
          from: center - newSpan / 2,
          to: center + newSpan / 2,
        });
      }
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ flexGrow: 1, height: '100vh', display: 'flex', flexDirection: 'column', bgcolor: COLORS.background, color: COLORS.textPrimary }}>
        <AppBar position="static" sx={{ bgcolor: COLORS.surface, backgroundImage: 'none', borderBottom: `1px solid ${COLORS.border}` }}>
          <Toolbar variant="dense">
            <ShowChartIcon sx={{ mr: 2, color: COLORS.accent }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontWeight: 500, fontSize: '1.1rem', letterSpacing: '0.02em', color: COLORS.textPrimary }}>
              Stocky
            </Typography>
            <Chip 
              label={isConnected ? "Connected" : "Disconnected"} 
              variant="outlined" 
              size="small"
              sx={{ 
                  mr: 1, height: 24, fontSize: '0.75rem', 
                  color: isConnected ? COLORS.success : COLORS.danger,
                  borderColor: isConnected ? 'rgba(38, 166, 154, 0.3)' : 'rgba(239, 83, 80, 0.3)',
                  bgcolor: isConnected ? 'rgba(38, 166, 154, 0.05)' : 'rgba(239, 83, 80, 0.05)'
              }}
            />
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ mt: 2, mb: 2, flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
          <Paper 
            elevation={0} 
            sx={{ 
              p: 2, 
              display: 'flex', 
              flexDirection: 'column', 
              height: '100%', 
              bgcolor: COLORS.surface, 
              borderRadius: 1,
              border: `1px solid ${COLORS.border}`,
            }}
          >
            
            <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 2 }}>
               <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                  <FormControl variant="outlined" size="small" sx={{ minWidth: 180 }}>
                      <InputLabel id="symbol-select-label" sx={{ color: COLORS.textSecondary, fontSize: '0.85rem' }}>Symbol</InputLabel>
                      <Select
                        labelId="symbol-select-label"
                        value={activeSymbol}
                        label="Symbol"
                        onChange={(e) => setActiveSymbol(e.target.value)}
                        sx={{ color: COLORS.textPrimary, fontSize: '0.85rem', '.MuiOutlinedInput-notchedOutline': { borderColor: COLORS.borderLight } }}
                        renderValue={(selected) => {
                          const meta = symbolMetadata[selected];
                          return (
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              {meta?.logo && <img src={meta.logo} style={{ width: 16, height: 16, borderRadius: '50%' }} />}
                              <Typography sx={{ fontSize: '0.85rem', fontWeight: 500 }}>
                                {meta ? meta.name : selected}
                              </Typography>
                            </Box>
                          );
                        }}
                      >
                        {availableSymbols.map((s) => {
                          const meta = symbolMetadata[s];
                          const exchange = s.split(':')[0];
                          return (
                            <MenuItem key={s} value={s} sx={{ py: 1, px: 1.5 }}>
                              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                                      {meta?.logo ? (
                                          <img src={meta.logo} style={{ width: 24, height: 24, borderRadius: '50%' }} />
                                      ) : (
                                          <Box sx={{ width: 24, height: 24, borderRadius: '50%', bgcolor: 'rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                              <ShowChartIcon sx={{ fontSize: 14, color: COLORS.textDisabled }} />
                                          </Box>
                                      )}
                                      <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                                          <Typography sx={{ fontSize: '0.85rem', fontWeight: 500, color: COLORS.textPrimary, lineHeight: 1.2 }}>
                                              {meta ? meta.name : s}
                                          </Typography>
                                          <Typography sx={{ fontSize: '0.7rem', color: COLORS.textSecondary, letterSpacing: '0.02em' }}>
                                              {s}
                                          </Typography>
                                      </Box>
                                  </Box>
                                  <Box sx={{ 
                                      fontSize: '0.65rem', 
                                      fontWeight: 600, 
                                      color: COLORS.textSecondary, 
                                      bgcolor: 'rgba(255,255,255,0.05)', 
                                      px: 0.8, py: 0.2, 
                                      borderRadius: 1,
                                      border: '1px solid rgba(255,255,255,0.1)'
                                  }}>
                                      {exchange}
                                  </Box>
                              </Box>
                            </MenuItem>
                          );
                        })}
                        {availableSymbols.length === 0 && <MenuItem disabled>Waiting for data...</MenuItem>}
                      </Select>
                  </FormControl>
                  
                  <FormControl variant="outlined" size="small" sx={{ minWidth: 120 }}>
                      <InputLabel id="timeframe-select-label" sx={{ color: COLORS.textSecondary, fontSize: '0.85rem' }}>Timeframe</InputLabel>
                      <Select
                        labelId="timeframe-select-label"
                        value={timeframe}
                        label="Timeframe"
                        onChange={(e) => setTimeframe(Number(e.target.value))}
                        sx={{ color: COLORS.textPrimary, fontSize: '0.85rem', '.MuiOutlinedInput-notchedOutline': { borderColor: COLORS.borderLight } }}
                      >
                        {AVAILABLE_TIMEFRAMES.map((tf) => (
                          <MenuItem key={tf.value} value={tf.value} sx={{ fontSize: '0.85rem' }}>{tf.label}</MenuItem>
                        ))}
                      </Select>
                  </FormControl>

                  <ToggleButtonGroup
                    value={chartType}
                    exclusive
                    onChange={(_, value) => value && setChartType(value)}
                    size="small"
                    aria-label="chart type"
                    sx={{ bgcolor: COLORS.border, height: 32 }}
                  >
                    <Tooltip title="Candlestick Chart">
                      <ToggleButton value="candlestick" sx={{ border: 'none', color: COLORS.textSecondary, '&.Mui-selected': { color: COLORS.accent, bgcolor: COLORS.borderLight } }}>
                        <BarChartIcon fontSize="small" />
                      </ToggleButton>
                    </Tooltip>
                    <Tooltip title="Line Chart">
                      <ToggleButton value="line" sx={{ border: 'none', color: COLORS.textSecondary, '&.Mui-selected': { color: COLORS.accent, bgcolor: COLORS.borderLight } }}>
                        <TimelineIcon fontSize="small" />
                      </ToggleButton>
                    </Tooltip>
                  </ToggleButtonGroup>

                  <Tooltip title={isLogScale ? "Switch to Linear Scale" : "Switch to Logarithmic Scale"}>
                    <ToggleButton
                      value="check"
                      selected={isLogScale}
                      onChange={() => setIsLogScale(!isLogScale)}
                      size="small"
                      sx={{ height: 32, color: COLORS.textSecondary, border: `1px solid ${COLORS.borderLight}`, '&.Mui-selected': { color: COLORS.accent, bgcolor: COLORS.borderLight, borderColor: COLORS.borderLight } }}
                    >
                      <Typography variant="caption" sx={{ fontWeight: 600, px: 0.5 }}>LOG</Typography>
                    </ToggleButton>
                  </Tooltip>

                  {!isAutoScale && (
                    <Button 
                      variant="text" 
                      size="small" 
                      onClick={handleResetScale}
                      sx={{ color: COLORS.accent, fontSize: '0.75rem', fontWeight: 600, minWidth: 'auto', px: 1, '&:hover': { bgcolor: 'rgba(240, 185, 11, 0.1)' } }}
                    >
                      RESET SCALE
                    </Button>
                  )}

                  <Box sx={{ display: 'flex', gap: 0.5, ml: 1 }}>
                    <Tooltip title="Zoom In">
                      <IconButton onClick={handleZoomIn} size="small" sx={{ border: `1px solid ${COLORS.borderLight}`, borderRadius: 1, color: COLORS.textPrimary, p: '4px' }}>
                        <ZoomInIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Zoom Out">
                      <IconButton onClick={handleZoomOut} size="small" sx={{ border: `1px solid ${COLORS.borderLight}`, borderRadius: 1, color: COLORS.textPrimary, p: '4px' }}>
                        <ZoomOutIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
               </Box>
            </Box>

            <Box 
              sx={{ 
                flexGrow: 1, 
                width: '100%', 
                position: 'relative',
                border: `1px solid ${COLORS.border}`,
                borderRadius: 1,
                overflow: 'hidden',
                bgcolor: COLORS.background
              }} 
            >
               <Box sx={{ position: 'absolute', top: 12, left: 12, zIndex: 10 }}>
                 <StockLegend 
                    symbol={activeSymbol}
                    metadata={symbolMetadata[activeSymbol]}
                    candle={legendData.candle}
                    mas={legendData.mas}
                 />
               </Box>
               <div ref={chartContainerRef} style={{ width: '100%', height: '100%' }} />
            </Box>
            
          </Paper>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
