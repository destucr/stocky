import React from 'react';
import { Box, Typography } from '@mui/material';
import { UI_COLORS } from '../theme';

interface StockLegendProps {
  symbol: string;
  metadata?: { name: string; logo: string };
  candle?: {
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  } | null;
  mas?: {
    ma7?: number;
    ma25?: number;
    ma99?: number;
  };
}

const formatVal = (val: number | null | undefined) => {
  if (val === undefined || val === null) return '-';
  if (val >= 1000) return val.toFixed(2);
  if (val >= 1) return val.toFixed(2);
  return val.toPrecision(5);
};

const DataField = ({ label, value, color }: { label: string; value: string; color?: string }) => (
  <Box sx={{ display: 'flex', flexDirection: 'column' }}>
    <Typography sx={{ color: UI_COLORS.textSecondary, fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.05em', mb: '1px' }}>
      {label}
    </Typography>
    <Typography sx={{ color: color || UI_COLORS.textPrimary, fontFamily: '"Roboto Mono", monospace', fontSize: '13px', fontWeight: 500 }}>
      {value}
    </Typography>
  </Box>
);

const MAField = ({ label, value, color }: { label: string; value: string; color: string }) => (
  <Box sx={{ display: 'flex', alignHover: 'center', gap: 0.5, fontSize: '11px' }}>
    <Box sx={{ width: 8, height: 2, bgcolor: color, borderRadius: '1px', mt: 1 }} />
    <Typography component="span" sx={{ color: UI_COLORS.textSecondary, fontSize: 'inherit' }}>{label}</Typography>
    <Typography component="span" sx={{ color: color, fontFamily: '"Roboto Mono", monospace', fontSize: 'inherit', fontWeight: 500 }}>{value}</Typography>
  </Box>
);

const StockLegend: React.FC<StockLegendProps> = ({ symbol, metadata, candle, mas }) => {
  if (!candle) return null;

  const isUp = candle.close >= candle.open;
  const priceColor = isUp ? UI_COLORS.success : UI_COLORS.danger;
  
  const volume = candle.volume || 0;
  const volStr = volume >= 1000 ? `${(volume / 1000).toFixed(2)}K` : volume.toFixed(0);

  return (
    <Box sx={{
      display: 'flex',
      flexDirection: 'column',
      gap: 1,
      background: UI_COLORS.overlayBg,
      backdropFilter: 'blur(8px)',
      p: '10px 14px',
      borderRadius: 1.5,
      border: '1px solid rgba(71, 77, 87, 0.3)',
      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
      minWidth: 'fit-content',
      userSelect: 'text',
      pointerEvents: 'none',
    }}>
      <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.2 }}>
          {metadata?.logo && (
            <Box component="img" src={metadata.logo} sx={{ width: 24, height: 24, borderRadius: '50%', objectFit: 'contain', bgcolor: 'rgba(255,255,255,0.05)' }} />
          )}
          <Box sx={{ display: 'flex', flexDirection: 'column' }}>
            <Typography sx={{ fontWeight: 600, fontSize: '14px', color: UI_COLORS.textPrimary, lineHeight: 1.2 }}>
              {metadata?.name || symbol}
            </Typography>
            <Typography sx={{ fontSize: '10px', color: UI_COLORS.textSecondary, letterSpacing: '0.02em' }}>
              {symbol}
            </Typography>
          </Box>
        </Box>

        <Box sx={{ display: 'flex', gap: 1.5 }}>
          <DataField label="Open" value={formatVal(candle.open)} color={priceColor} />
          <DataField label="High" value={formatVal(candle.high)} color={priceColor} />
          <DataField label="Low" value={formatVal(candle.low)} color={priceColor} />
          <DataField label="Close" value={formatVal(candle.close)} color={priceColor} />
          <DataField label="Volume" value={volStr} />
        </Box>
      </Box>

      <Box sx={{ display: 'flex', gap: 1.5, pt: 0.5, borderTop: `1px solid rgba(71, 77, 87, 0.2)` }}>
        <MAField label="MA(7)" value={formatVal(mas?.ma7)} color={UI_COLORS.ma7} />
        <MAField label="MA(25)" value={formatVal(mas?.ma25)} color={UI_COLORS.ma25} />
        <MAField label="MA(99)" value={formatVal(mas?.ma99)} color={UI_COLORS.ma99} />
      </Box>
    </Box>
  );
};

export default StockLegend;
