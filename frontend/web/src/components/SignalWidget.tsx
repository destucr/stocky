import { useState, useEffect } from 'react';
import { Box, Typography, Tooltip } from '@mui/material';
import { UI_COLORS as COLORS } from '../theme';

interface Signal {
  type: string;
  price: number;
  target: number;
  stop: number;
  probability: number;
  confidence?: string;
  timestamp: string;
  rsi?: number;
  trend?: string;
  reason?: string;
}

interface SignalWidgetProps {
  symbol: string;
}

const SignalWidget = ({ symbol }: SignalWidgetProps) => {
  const [signal, setSignal] = useState<Signal | null>(null);

  useEffect(() => {
    const fetchSignal = async () => {
      try {
        const response = await fetch(`http://localhost:8080/api/signal?symbol=${encodeURIComponent(symbol)}`);
        if (response.ok) {
          const data = await response.json();
          if (data && data.type) {
            setSignal(data);
          } else {
            setSignal(null);
          }
        }
      } catch (err) {
        console.error("Failed to fetch signal", err);
      }
    };

    if (symbol) {
        fetchSignal();

        const handleSignalEvent = (event: any) => {
            const newSignal = event.detail;
            const isMatch = newSignal.symbol === symbol || 
                           symbol.endsWith(newSignal.symbol) || 
                           newSignal.symbol.endsWith(symbol);
            if (isMatch) {
                setSignal(newSignal);
            }
        };

        window.addEventListener('stocky-signal', handleSignalEvent);
        const interval = setInterval(fetchSignal, 60000);
        return () => {
            clearInterval(interval);
            window.removeEventListener('stocky-signal', handleSignalEvent);
        };
    }
  }, [symbol]);

  if (!signal) return null;

  const isBuy = signal.type === 'BUY';
  const isSell = signal.type === 'SELL';
  const hasAction = isBuy || isSell;
  
  const signalColor = isBuy ? COLORS.success : isSell ? COLORS.danger : COLORS.textSecondary;
  
  // Format price
  const formatPrice = (price: number) => {
    if (price >= 1000) return '$' + price.toLocaleString(undefined, { maximumFractionDigits: 0 });
    if (price >= 1) return '$' + price.toFixed(2);
    return '$' + price.toFixed(4);
  };

  // RSI indicator
  const rsiColor = signal.rsi ? (signal.rsi < 30 ? COLORS.success : signal.rsi > 70 ? COLORS.danger : COLORS.textSecondary) : COLORS.textSecondary;

  // Build tooltip content
  const tooltipContent = (
    <Box sx={{ p: 0.5 }}>
      <Typography sx={{ fontSize: 11, fontWeight: 600, mb: 0.5 }}>RSI Strategy Signal</Typography>
      {signal.reason && <Typography sx={{ fontSize: 10, opacity: 0.8 }}>{signal.reason}</Typography>}
      {hasAction && (
        <Box sx={{ mt: 1, fontSize: 10 }}>
          <Box>Entry: {formatPrice(signal.price)}</Box>
          <Box sx={{ color: COLORS.success }}>Take Profit: {formatPrice(signal.target)}</Box>
          <Box sx={{ color: COLORS.danger }}>Stop Loss: {formatPrice(signal.stop)}</Box>
        </Box>
      )}
    </Box>
  );

  return (
    <Tooltip title={tooltipContent} arrow placement="bottom">
      <Box sx={{ 
        display: 'flex', 
        alignItems: 'center', 
        gap: 1.5,
        cursor: 'default',
        userSelect: 'none'
      }}>
        {/* RSI reading */}
        {signal.rsi !== undefined && (
          <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 0.4 }}>
            <Typography sx={{ fontSize: 10, color: COLORS.textSecondary, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              RSI
            </Typography>
            <Typography sx={{ 
              fontSize: 13, 
              fontWeight: 600, 
              fontFamily: 'monospace',
              color: rsiColor
            }}>
              {signal.rsi.toFixed(0)}
            </Typography>
          </Box>
        )}

        {/* Separator dot */}
        <Box sx={{ width: 3, height: 3, borderRadius: '50%', bgcolor: COLORS.border }} />

        {/* Signal action */}
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: 0.5
        }}>
          {/* Pulsing dot for active signals */}
          {hasAction && (
            <Box sx={{ 
              width: 6, 
              height: 6, 
              borderRadius: '50%', 
              bgcolor: signalColor,
              animation: 'pulse 2s infinite',
              '@keyframes pulse': {
                '0%, 100%': { opacity: 1 },
                '50%': { opacity: 0.4 }
              }
            }} />
          )}
          <Typography sx={{ 
            fontSize: 12, 
            fontWeight: 600, 
            color: signalColor,
            letterSpacing: '0.02em'
          }}>
            {isBuy ? 'BUY' : isSell ? 'SELL' : 'HOLD'}
          </Typography>
        </Box>

        {/* Target price for actionable signals */}
        {hasAction && (
          <>
            <Box sx={{ width: 3, height: 3, borderRadius: '50%', bgcolor: COLORS.border }} />
            <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 0.4 }}>
              <Typography sx={{ fontSize: 10, color: COLORS.success, textTransform: 'uppercase' }}>
                TP
              </Typography>
              <Typography sx={{ fontSize: 11, fontFamily: 'monospace', color: COLORS.textPrimary }}>
                {formatPrice(signal.target)}
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 0.4 }}>
              <Typography sx={{ fontSize: 10, color: COLORS.danger, textTransform: 'uppercase' }}>
                SL
              </Typography>
              <Typography sx={{ fontSize: 11, fontFamily: 'monospace', color: COLORS.textPrimary }}>
                {formatPrice(signal.stop)}
              </Typography>
            </Box>
          </>
        )}
      </Box>
    </Tooltip>
  );
};

export default SignalWidget;
