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
import BarChartIcon from '@mui/icons-material/BarChart';
import TimelineIcon from '@mui/icons-material/Timeline';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import CenterFocusStrongIcon from '@mui/icons-material/CenterFocusStrong';
import theme, { UI_COLORS as COLORS } from './theme';
import StockLegend, { StockLegendRef } from './components/StockLegend';

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

const AVAILABLE_INTERVALS = [
  { label: '1 Second', value: 1 },
  { label: '5 Seconds', value: 5 },
  { label: '10 Seconds', value: 10 },
  { label: '1 Minute', value: 60 },
  { label: '5 Minutes', value: 300 },
  { label: '1 Hour', value: 3600 },
  { label: '1 Day', value: 86400 },
];

function App() {
  const [activeSymbol, setActiveSymbol] = useState<string>('');
  const [availableSymbols, setAvailableSymbols] = useState<string[]>([]);
  const [symbolMetadata, setSymbolMetadata] = useState<Record<string, SymbolMetadata>>({});
  const symbolMetadataRef = useRef<Record<string, SymbolMetadata>>({});
  const [isConnected, setIsConnected] = useState(false);
  const [chartType, setChartType] = useState<'candlestick' | 'line' | 'area'>('candlestick');
  const [isLogScale, setIsLogScale] = useState(false);
  const [chartInterval, setChartInterval] = useState<number>(60);
  const intervalRef = useRef(chartInterval);
  const lastSymbolRef = useRef<string>('');
  const [isAutoScale, setIsAutoScale] = useState(true);
  const lastPriceRef = useRef<number | null>(null);
  const isAutoScaleRef = useRef(true);
  const lastRangeRef = useRef<{ from: number, to: number } | null>(null);
  
  const [brokenLogos, setBrokenLogos] = useState<Record<string, boolean>>({});
  const legendRef = useRef<StockLegendRef>(null);

  const chartContainerRef = useRef<HTMLDivElement>(null);
  const volumeContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const volumeChartRef = useRef<IChartApi | null>(null);
  const isSyncingRef = useRef(false);
  
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const lineSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const areaSeriesRef = useRef<ISeriesApi<"Area"> | null>(null);
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

  // Sync intervalRef
  useEffect(() => {
    intervalRef.current = chartInterval;
  }, [chartInterval]);

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

  // Legend Updater - Now uses Ref for performance
  const updateLegendUI = (candle: any, _symbol: string, ma7?: number, ma25?: number, ma99?: number) => {
    legendRef.current?.update({
        candle,
        mas: { ma7, ma25, ma99 }
    });
  };

    // Initialize Charts
    useEffect(() => {
      if (chartContainerRef.current && volumeContainerRef.current) {
        const unsubscribes: (() => void)[] = [];
  
        const commonOptions = {                        layout: {
                          background: { type: ColorType.Solid, color: COLORS.background },
                          textColor: COLORS.textSecondary,
                          // Eliminate internal library padding
                          padding: { top: 0, bottom: 0, left: 0, right: 0 },
                          attributionLogo: false,
                        },
        grid: {
          vertLines: { color: COLORS.border },
          horzLines: { color: COLORS.border },
        },
        handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: true },
        handleScroll: { 
            mouseWheel: true, 
            pressedMouseMove: true,
            horzTouchDrag: true,
            vertTouchDrag: true,
        },
      };

      // 1. Price Chart (Background Layer)
      const priceChart = createChart(chartContainerRef.current, {
        ...commonOptions,
        width: chartContainerRef.current.clientWidth,
        height: chartContainerRef.current.clientHeight || 400,
        timeScale: { 
            visible: false, 
            borderVisible: false,
            shiftVisibleRangeOnNewBar: true,
            rightOffset: 150,
            barSpacing: 10,
        },
        rightPriceScale: { 
            borderColor: COLORS.borderLight, 
            autoScale: true, 
            scaleMargins: { top: 0.1, bottom: 0 }, // No margin at bottom
            minimumWidth: 80, // Fixed width to ensure alignment
        },
      });

      // 2. Volume Chart (Foreground Layer)
      const volumeChart = createChart(volumeContainerRef.current, {
        ...commonOptions,
        width: volumeContainerRef.current.clientWidth,
        height: 120,
        timeScale: {
          timeVisible: true,
          secondsVisible: true,
          borderColor: COLORS.borderLight,
          shiftVisibleRangeOnNewBar: true, // Auto-scroll to follow new data
          rightOffset: 150, // Large buffer on the right
          barSpacing: 10,
        },
        rightPriceScale: {
          borderColor: COLORS.borderLight,
          autoScale: true,
          scaleMargins: { top: 0, bottom: 0 }, // Content fills the pane
          minimumWidth: 80, // Matches price chart for perfect alignment
        },
      });

      // --- Bidirectional Synchronization ---
      const priceTimeScale = priceChart.timeScale();
      const volumeTimeScale = volumeChart.timeScale();

      const sync = (src: any, target: any) => {
        const handler = () => {
          if (isSyncingRef.current) return;
          isSyncingRef.current = true;
          try {
            const logicalRange = src.getVisibleLogicalRange();
            if (logicalRange) {
              target.setVisibleLogicalRange(logicalRange);
            }
          } catch (err) {
            // Ignore sync errors during load
          } finally {
            isSyncingRef.current = false;
          }
        };
        src.subscribeVisibleTimeRangeChange(handler);
        return () => src.unsubscribeVisibleTimeRangeChange(handler);
      };

      unsubscribes.push(sync(priceTimeScale, volumeTimeScale));
      unsubscribes.push(sync(volumeTimeScale, priceTimeScale));

              const candlestickSeries = priceChart.addCandlestickSeries({
                upColor: COLORS.success, 
                downColor: COLORS.danger, 
                borderVisible: false, 
                wickUpColor: COLORS.success, 
                wickDownColor: COLORS.danger, 
                visible: chartType === 'candlestick',
                priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
                lastValueVisible: true,
                priceLineVisible: true,
                priceLineSource: 1, // LastVisible source
                priceLineColor: COLORS.textPrimary,
                priceLineWidth: 1,
              });
      
              const lineSeries = priceChart.addLineSeries({
                color: COLORS.accent,
                lineWidth: 2,
                visible: chartType === 'line',
                lastValueVisible: true,
                priceLineVisible: true,
                priceLineColor: COLORS.accent,
                priceLineWidth: 1,
              });
      
              const areaSeries = priceChart.addAreaSeries({
                lineColor: COLORS.accent,
                topColor: 'rgba(33, 150, 243, 0.4)',
                bottomColor: 'rgba(33, 150, 243, 0.0)',
                lineWidth: 2,
                visible: chartType === 'area',
                lastValueVisible: true,
                priceLineVisible: true,
                priceLineColor: COLORS.accent,
                priceLineWidth: 1,
              });
      const volumeSeries = volumeChart.addHistogramSeries({
        color: COLORS.success,
        priceFormat: { type: 'volume' },
      });

      const m7 = priceChart.addLineSeries({ color: COLORS.ma7, lineWidth: 1 });
      const m25 = priceChart.addLineSeries({ color: COLORS.ma25, lineWidth: 1 });
      const m99 = priceChart.addLineSeries({ color: COLORS.ma99, lineWidth: 1 });

      chartRef.current = priceChart;
      volumeChartRef.current = volumeChart;
      seriesRef.current = candlestickSeries;
      lineSeriesRef.current = lineSeries;
      areaSeriesRef.current = areaSeries;
      volumeSeriesRef.current = volumeSeries;
      ma7SeriesRef.current = m7;
      ma25SeriesRef.current = m25;
      ma99SeriesRef.current = m99;

      const handleCrosshair = (param: MouseEventParams) => {
        if (param.point && seriesRef.current) {
          lastPriceRef.current = seriesRef.current.coordinateToPrice(param.point.y);
        }

        const validPoint = param.time && param.point && param.point.x >= 0;
        if (validPoint) {
            const cData: any = param.seriesData.get(candlestickSeries);
            const lData: any = param.seriesData.get(lineSeries);
            const m7d: any = param.seriesData.get(m7);
            const m25d: any = param.seriesData.get(m25);
            const m99d: any = param.seriesData.get(m99);
            
            if (cData || lData) {
                const data = cData || { open: lData.value, high: lData.value, low: lData.value, close: lData.value };
                updateLegendUI({ ...data, volume: cData?.volume }, activeSymbolRef.current, m7d?.value, m25d?.value, m99d?.value);
                return;
            }
        }
        
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

      priceChart.subscribeCrosshairMove(handleCrosshair);
      unsubscribes.push(() => priceChart.unsubscribeCrosshairMove(handleCrosshair));

      // --- Bidirectional Crosshair Sync ---
      const syncCrosshair = (sourceChart: IChartApi, targetChart: IChartApi, targetSeries: ISeriesApi<any>) => {
        const handler = (param: MouseEventParams) => {
          if (isSyncingRef.current) return;
          isSyncingRef.current = true;
          try {
            if (!param.time) {
              targetChart.clearCrosshairPosition();
            } else {
              targetChart.setCrosshairPosition(0, param.time, targetSeries);
            }
          } catch (err) {
            // Ignore crosshair sync errors
          } finally {
            isSyncingRef.current = false;
          }
        };
        sourceChart.subscribeCrosshairMove(handler);
        return () => sourceChart.unsubscribeCrosshairMove(handler);
      };

      if (seriesRef.current && volumeSeriesRef.current) {
        unsubscribes.push(syncCrosshair(priceChart, volumeChart, volumeSeriesRef.current));
        unsubscribes.push(syncCrosshair(volumeChart, priceChart, seriesRef.current));
      }

      // --- Real-time Vertical Price Zoom Logic ---
      const handleWheel = (e: WheelEvent) => {
        if (!chartRef.current || !seriesRef.current || !container) return;
        
        // Only perform vertical zoom if Alt key is pressed
        if (!e.altKey) return;
        
        e.preventDefault();

        const priceScale = chartRef.current.priceScale('right');
        // @ts-ignore
        const currentOptions = priceScale.options();
        // @ts-ignore
        const range = (currentOptions && currentOptions.priceRange) || lastRangeRef.current;
        
        if (!range) return;

        const rect = container.getBoundingClientRect();
        const focusPrice = seriesRef.current.coordinateToPrice(e.clientY - rect.top);
        if (focusPrice === null) return;

        const factor = Math.pow(1.0005, e.deltaY * (e.deltaMode === 1 ? 20 : 1));
        const newRangeSize = (range.to - range.from) * factor;

        if (newRangeSize < 0.001 || newRangeSize > 10000000) return;

        const relativePos = (focusPrice - range.from) / (range.to - range.from);
        const newRange = { from: focusPrice - relativePos * newRangeSize, to: focusPrice + (1 - relativePos) * newRangeSize };

        lastRangeRef.current = newRange;

        if (isAutoScaleRef.current) {
          isAutoScaleRef.current = false;
          setTimeout(() => setIsAutoScale(false), 0);
        }

        priceScale.applyOptions({ 
            autoScale: false,
            // @ts-ignore
            priceRange: newRange 
        });
      };

      // --- Vertical Panning Logic ---
      let isPanning = false;
      let panStartRange: any = null;
      let panStartY = 0;

      const handleMouseDown = (e: MouseEvent) => {
        if (e.button !== 0 || !chartRef.current || !container) return;
        const priceScale = chartRef.current.priceScale('right');
        const options: any = priceScale.options();
        panStartRange = options.priceRange || lastRangeRef.current;
        if (!panStartRange) return;
        
        isPanning = true;
        panStartY = e.clientY;
      };

      const handleMouseMove = (e: MouseEvent) => {
        if (!isPanning || !panStartRange || !chartRef.current || !container) return;
        
        const deltaY = e.clientY - panStartY;
        if (Math.abs(deltaY) < 1) return;

        const chartHeight = container.clientHeight || 400;
        const rangeHeight = panStartRange.to - panStartRange.from;
        const priceDelta = (deltaY / chartHeight) * rangeHeight;

        const newRange = {
          from: panStartRange.from + priceDelta,
          to: panStartRange.to + priceDelta,
        };

        chartRef.current.priceScale('right').applyOptions({
          autoScale: false,
          // @ts-ignore
          priceRange: newRange,
        });
        
        lastRangeRef.current = newRange;
        if (isAutoScaleRef.current) {
          isAutoScaleRef.current = false;
          setTimeout(() => setIsAutoScale(false), 0);
        }
      };

      const handleMouseUp = () => {
        isPanning = false;
        panStartRange = null;
      };

      const container = chartContainerRef.current;
      if (container) {
        container.addEventListener('wheel', handleWheel, { passive: false });
        container.addEventListener('mousedown', handleMouseDown);
      }
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);

      const handleResize = () => {
        if (chartRef.current && volumeChartRef.current && chartContainerRef.current) {
          chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
          volumeChartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
        }
      };

              window.addEventListener('resize', handleResize);
              return () => {
                window.removeEventListener('resize', handleResize);
                window.removeEventListener('mousemove', handleMouseMove);
                window.removeEventListener('mouseup', handleMouseUp);
                if (container) {
                    container.removeEventListener('wheel', handleWheel);
                    container.removeEventListener('mousedown', handleMouseDown);
                }
                unsubscribes.forEach(u => u());
                priceChart.remove();
                volumeChart.remove();
              };    }
  }, []);

  // Handle Chart Type Change
  useEffect(() => {
    if (seriesRef.current && lineSeriesRef.current && areaSeriesRef.current) {
      seriesRef.current.applyOptions({ visible: chartType === 'candlestick' });
      lineSeriesRef.current.applyOptions({ visible: chartType === 'line' });
      areaSeriesRef.current.applyOptions({ visible: chartType === 'area' });
    }
  }, [chartType]);

  // Handle Scale Change
  useEffect(() => {
    if (chartRef.current) {
      // Clear manual range tracking when switching modes to avoid coordinate mismatches
      lastRangeRef.current = null;
      setIsAutoScale(true);
      isAutoScaleRef.current = true;

      chartRef.current.priceScale('right').applyOptions({
        autoScale: true,
        mode: isLogScale ? PriceScaleMode.Logarithmic : PriceScaleMode.Normal,
      });
      
      // Force a slight delay and re-apply autoScale to ensure candles are re-plotted
      setTimeout(() => {
        chartRef.current?.priceScale('right').applyOptions({ autoScale: true });
      }, 50);
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
          // Finnhub sometimes sends multiple JSON objects in one frame separated by newlines
          const raw = event.data.toString();
          const parts = raw.split('\n').filter((p: string) => p.trim().length > 0);

          parts.forEach((jsonStr: string) => {
            try {
              const message: BackendMessage = JSON.parse(jsonStr);
              if (message.type === 'trade') {
                const receivedSymbols = new Set(message.data.map(t => t.s));
                setAvailableSymbols(prev => {
                  const newSymbols = Array.from(receivedSymbols).filter(s => !prev.includes(s));
                  return newSymbols.length > 0 ? [...prev, ...newSymbols].sort() : prev;
                });

                // Process all trades in the message but only update chart for the active symbol
                const trades = message.data.filter(t => t.s === activeSymbolRef.current);
                if (trades.length > 0) {
                    trades.forEach(trade => processTrade(trade));
                }
              }
            } catch (innerErr) {
              console.error("Single frame parse error:", innerErr, "Raw string:", jsonStr);
            }
          });
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
        // Reset building state when symbol or interval changes
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
            areaSeriesRef.current?.setData([]);
            volumeSeriesRef.current?.setData([]);
            ma7SeriesRef.current?.setData([]);
            ma25SeriesRef.current?.setData([]);
            ma99SeriesRef.current?.setData([]);
          }
          lastSymbolRef.current = activeSymbol;
        }
    
            // Always fetch new history for the selected interval
            if (activeSymbol) {
              fetchHistory(activeSymbol, chartInterval);
            }
          }, [activeSymbol, chartInterval]);  const handleResetScale = () => {
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
          // Request more candles for smaller intervals to keep the chart full
          let limit = 500;      if (interval <= 1) limit = 2000;
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
        let lastT = -1;

        data.forEach((c: any) => {
          if (!c || c.time === undefined || c.time === null || c.close === undefined || c.close === null) {
             return;
          }
          
          const t = Math.floor(new Date(c.time).getTime() / 1000) as Time;
          const close = Number(c.close);
          
          if (isNaN(t as number) || isNaN(close) || (t as number) <= lastT) {
              return;
          }
          
          lastT = t as number;
          
          const open = typeof c.open === 'number' ? c.open : close;
          const high = typeof c.high === 'number' ? c.high : close;
          const low = typeof c.low === 'number' ? c.low : close;
          const volume = typeof c.volume === 'number' ? c.volume : 0;
          
          candles.push({ time: t, open, high, low, close, volume });
          lineData.push({ time: t, value: close });
          volumeData.push({ 
            time: t, 
            value: volume, 
            color: (close >= open) ? COLORS.success : COLORS.danger 
          });
        });

        if (candles.length === 0) {
            console.warn("No valid candles after processing data");
            return;
        }

        console.log(`History fetched for ${symbol}: ${candles.length} candles`);
        
        // We set candleHistoryRef to all candles EXCEPT the last one, 
        // because the last one is stored in currentCandleRef and will be 
        // pushed to history only when it is finalized (i.e., when a new candle starts).
        candleHistoryRef.current = candles.slice(0, -1);
        
        const ma7 = calculateMA(candles, 7);
        const ma25 = calculateMA(candles, 25);
        const ma99 = calculateMA(candles, 99);

        ma7HistoryRef.current = ma7;
        ma25HistoryRef.current = ma25;
        ma99HistoryRef.current = ma99;

        // Disable sync while loading new data
        isSyncingRef.current = true;
        try {
            // Update series with new resolution
            seriesRef.current?.setData(candles);
            lineSeriesRef.current?.setData(lineData);
            areaSeriesRef.current?.setData(lineData);
            volumeSeriesRef.current?.setData(volumeData);
            ma7SeriesRef.current?.setData(ma7);
            ma25SeriesRef.current?.setData(ma25);
            ma99SeriesRef.current?.setData(ma99);

            // Snap to latest data
            chartRef.current?.timeScale().scrollToRealTime();
        } catch (setDataErr) {
            console.error("Error setting chart data:", setDataErr);
            throw setDataErr;
        } finally {
            // Re-enable sync after a small delay to let chart stabilize
            setTimeout(() => { isSyncingRef.current = false; }, 50);
        }

        const latestCandle = candles[candles.length - 1];
        lastCandleTimeRef.current = latestCandle.time as number;
        currentCandleRef.current = {
          time: latestCandle.time,
          open: latestCandle.open, high: latestCandle.high, low: latestCandle.low, close: latestCandle.close,
          volume: latestCandle.volume
        };
        updateLegendUI(
            currentCandleRef.current, 
            symbol,
            ma7.length > 0 ? ma7[ma7.length - 1].value : undefined,
            ma25.length > 0 ? ma25[ma25.length - 1].value : undefined,
            ma99.length > 0 ? ma99[ma99.length - 1].value : undefined
        );

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
        // Disable sync while clearing data
        isSyncingRef.current = true;
        try {
            // Clear if no data
            seriesRef.current?.setData([]);
            lineSeriesRef.current?.setData([]);
            volumeSeriesRef.current?.setData([]);
            ma7SeriesRef.current?.setData([]);
            ma25SeriesRef.current?.setData([]);
            ma99SeriesRef.current?.setData([]);
            lastRangeRef.current = null;
        } finally {
            isSyncingRef.current = false;
        }
      }
    } catch (err) { console.error("History Error:", err); }
  };

  const processTrade = (trade: TradeData) => {
    const tradeTime = Math.floor(trade.t / 1000);
    const candleIntervalVal = intervalRef.current; 
    const candleTime = Math.floor(tradeTime / candleIntervalVal) * candleIntervalVal;

    if (candleTime < lastCandleTimeRef.current) return;

    const isNewCandle = !currentCandleRef.current || candleTime > lastCandleTimeRef.current;

    if (isNewCandle) {
      if (currentCandleRef.current) {
        candleHistoryRef.current.push({ 
            time: currentCandleRef.current.time, 
            close: currentCandleRef.current.close 
        });
        if (candleHistoryRef.current.length > 1000) candleHistoryRef.current.shift();
      }

      currentCandleRef.current = {
        time: candleTime as Time,
        open: trade.p, high: trade.p, low: trade.p, close: trade.p,
        volume: trade.v,
      };
      lastCandleTimeRef.current = candleTime;
    } else {
      const c = currentCandleRef.current!;
      c.high = Math.max(c.high, trade.p);
      c.low = Math.min(c.low, trade.p);
      c.close = trade.p;
      c.volume += trade.v;
    }

    const v7 = calculateLastMA(7, trade.p);
    const v25 = calculateLastMA(25, trade.p);
    const v99 = calculateLastMA(99, trade.p);

    const time = candleTime as Time;
    const price = trade.p;
    
    // Instant Chart Update
    if (chartType === 'candlestick') {
        seriesRef.current?.update(currentCandleRef.current!);
    } else if (chartType === 'line') {
        lineSeriesRef.current?.update({ time, value: price });
    } else {
        areaSeriesRef.current?.update({ time, value: price });
    }

    volumeSeriesRef.current?.update({
        time, value: currentCandleRef.current!.volume,
        color: currentCandleRef.current!.close >= currentCandleRef.current!.open ? COLORS.success : COLORS.danger
    });

    if (v7 !== null) ma7SeriesRef.current?.update({ time, value: v7 });
    if (v25 !== null) ma25SeriesRef.current?.update({ time, value: v25 });
    if (v99 !== null) ma99SeriesRef.current?.update({ time, value: v99 });

    updateLegendUI(currentCandleRef.current, activeSymbolRef.current, v7 ?? undefined, v25 ?? undefined, v99 ?? undefined);
  };

  const calculateLastMA = (period: number, currentPrice: number) => {
      const hist = candleHistoryRef.current;
      const len = hist.length;
      if (len + 1 < period) return null;
      
      let sum = currentPrice;
      for (let i = 0; i < period - 1; i++) {
          sum += hist[len - 1 - i].close;
      }
      return sum / period;
  };

      const handleZoomIn = () => {
        const currentIndex = AVAILABLE_INTERVALS.findIndex(tf => tf.value === chartInterval);
        if (currentIndex > 0) {
          setChartInterval(AVAILABLE_INTERVALS[currentIndex - 1].value);
        } else if (chartRef.current) {
          const timeScale = chartRef.current.timeScale();
          const logicalRange = timeScale.getVisibleLogicalRange();
          if (logicalRange) {
            const span = logicalRange.to - logicalRange.from;
            const center = (logicalRange.to + logicalRange.from) / 2;
            const newSpan = span * 0.9; 
            timeScale.setVisibleLogicalRange({
              from: center - newSpan / 2,
              to: center + newSpan / 2,
            });
          }
        }
      };
  
      const handleZoomOut = () => {
        const currentIndex = AVAILABLE_INTERVALS.findIndex(tf => tf.value === chartInterval);
        if (currentIndex < AVAILABLE_INTERVALS.length - 1) {
          setChartInterval(AVAILABLE_INTERVALS[currentIndex + 1].value);
        } else if (chartRef.current) {
          const timeScale = chartRef.current.timeScale();
          const logicalRange = timeScale.getVisibleLogicalRange();
          if (logicalRange) {
            const span = logicalRange.to - logicalRange.from;
            const center = (logicalRange.to + logicalRange.from) / 2;
            const newSpan = span * 1.1; 
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
            <TimelineIcon sx={{ mr: 1.5, color: COLORS.accent, fontSize: 20 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontWeight: 500, fontSize: '1rem', letterSpacing: '0.01em', color: COLORS.textPrimary }}>
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
                          const isLogoBroken = brokenLogos[s];

                          return (
                            <MenuItem key={s} value={s} sx={{ py: 1, px: 1.5 }}>
                              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                                      {meta?.logo && !isLogoBroken ? (
                                          <img 
                                            src={meta.logo} 
                                            onError={() => setBrokenLogos(prev => ({ ...prev, [s]: true }))}
                                            style={{ width: 20, height: 20, borderRadius: '50%', filter: 'saturate(0.8)' }} 
                                          />
                                      ) : (
                                          <Box sx={{ width: 20, height: 20, borderRadius: '50%', bgcolor: 'rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                              <Typography sx={{ fontSize: '10px', fontWeight: 700, color: COLORS.textSecondary }}>{s.substring(0, 1)}</Typography>
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
                      <InputLabel id="interval-select-label" sx={{ color: COLORS.textSecondary, fontSize: '0.85rem' }}>Interval</InputLabel>
                      <Select
                        labelId="interval-select-label"
                        value={chartInterval}
                        label="Interval"
                        onChange={(e) => setChartInterval(Number(e.target.value))}
                        sx={{ color: COLORS.textPrimary, fontSize: '0.85rem', '.MuiOutlinedInput-notchedOutline': { borderColor: COLORS.borderLight } }}
                      >
                        {AVAILABLE_INTERVALS.map((tf) => (
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
                    <Tooltip title="Area Chart">
                      <ToggleButton value="area" sx={{ border: 'none', color: COLORS.textSecondary, '&.Mui-selected': { color: COLORS.accent, bgcolor: COLORS.borderLight } }}>
                        <ShowChartIcon fontSize="small" />
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
                                        <Tooltip title="Reset chart scale">
                                          <Button 
                                            variant="outlined" 
                                            size="small" 
                                            startIcon={<CenterFocusStrongIcon sx={{ fontSize: '1rem !important' }} />}
                                            onClick={handleResetScale}
                                            sx={{ 
                                              color: COLORS.textPrimary, 
                                              borderColor: COLORS.borderLight,
                                              fontSize: '0.75rem', 
                                              fontWeight: 500, 
                                              height: 32,
                                              px: 1.5,
                                              '&:hover': { 
                                                  bgcolor: 'rgba(255, 255, 255, 0.05)',
                                                  borderColor: COLORS.accent,
                                                  color: COLORS.accent
                                              } 
                                            }}
                                          >
                                            Reset scale
                                          </Button>
                                        </Tooltip>
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

            {/* Main Chart Area - Single Screen Viewport */}
            <Box 
              sx={{ 
                flexGrow: 1, 
                width: '100%', 
                display: 'flex',
                flexDirection: 'column',
                position: 'relative',
                overflow: 'hidden',
                bgcolor: COLORS.background,
              }} 
            >
                {/* Price Chart Wrapper */}
                <Box 
                  sx={{ 
                      flexGrow: 1,
                      width: '100%', 
                      position: 'relative',
                      zIndex: 1,
                  }}
                >
                    {/* Legend Overlay */}
                    <Box sx={{ position: 'absolute', top: 12, left: 12, zIndex: 10 }}>
                      <StockLegend 
                        ref={legendRef}
                        symbol={activeSymbol}
                        metadata={symbolMetadata[activeSymbol]}
                      />
                    </Box>
                    <Box ref={chartContainerRef} sx={{ width: '100%', height: '100%' }} />
                </Box>

                {/* Volume Pane */}
                <Box 
                  sx={{ 
                      width: '100%', 
                      height: 120, 
                      zIndex: 2,
                      bgcolor: COLORS.surface,
                      borderTop: `1px solid ${COLORS.border}`,
                      boxShadow: '0 -10px 30px rgba(0,0,0,0.4)',
                      overflow: 'hidden'
                  }}
                >
                    <Box ref={volumeContainerRef} sx={{ width: '100%', height: '100%' }} />
                </Box>
            </Box>

            <Box sx={{ mt: 1, display: 'flex', justifyContent: 'flex-end' }}>
              <Typography 
                variant="caption" 
                sx={{ 
                  color: COLORS.textSecondary, 
                  fontSize: '0.65rem',
                  opacity: 0.7,
                  '& a': { 
                    color: COLORS.accent, 
                    textDecoration: 'none',
                    '&:hover': { textDecoration: 'underline' }
                  } 
                }}
              >
                Charts by <a href="https://www.tradingview.com/" target="_blank" rel="noopener noreferrer">TradingView</a>
              </Typography>
            </Box>
            
          </Paper>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
