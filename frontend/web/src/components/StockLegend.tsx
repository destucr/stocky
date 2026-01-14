import React, { useImperativeHandle, useRef, forwardRef } from 'react';
import { Box, Typography } from '@mui/material';
import { UI_COLORS } from '../theme';

interface StockLegendProps {
  symbol: string;
  metadata?: { name: string; logo: string };
  initialCandle?: any;
  initialMas?: any;
}

export interface StockLegendRef {
  update: (data: { candle: any; mas: any }) => void;
}

const formatVal = (val: number | null | undefined) => {
  if (val === undefined || val === null) return '-';
  if (val >= 1000) return val.toFixed(2);
  if (val >= 1) return val.toFixed(2);
  return val.toPrecision(5);
};

const StockLegend = forwardRef<StockLegendRef, StockLegendProps>(({ symbol, metadata }, ref) => {
  const openRef = useRef<HTMLElement>(null);
  const highRef = useRef<HTMLElement>(null);
  const lowRef = useRef<HTMLElement>(null);
  const closeRef = useRef<HTMLElement>(null);
  const volRef = useRef<HTMLElement>(null);
  const ma7Ref = useRef<HTMLElement>(null);
  const ma25Ref = useRef<HTMLElement>(null);
  const ma99Ref = useRef<HTMLElement>(null);
  const [imgError, setImgError] = React.useState(false);

  useImperativeHandle(ref, () => ({
    update: ({ candle, mas }) => {
      if (!candle) return;
      const isUp = candle.close >= candle.open;
      const color = isUp ? UI_COLORS.success : UI_COLORS.danger;
      
      if (openRef.current) { openRef.current.innerText = formatVal(candle.open); openRef.current.style.color = color; }
      if (highRef.current) { highRef.current.innerText = formatVal(candle.high); highRef.current.style.color = color; }
      if (lowRef.current) { lowRef.current.innerText = formatVal(candle.low); lowRef.current.style.color = color; }
      if (closeRef.current) { closeRef.current.innerText = formatVal(candle.close); closeRef.current.style.color = color; }
      
      if (volRef.current) {
        const v = candle.volume || 0;
        volRef.current.innerText = v >= 1000 ? `${(v / 1000).toFixed(2)}K` : v.toFixed(0);
      }

      if (ma7Ref.current) ma7Ref.current.innerText = formatVal(mas?.ma7);
      if (ma25Ref.current) ma25Ref.current.innerText = formatVal(mas?.ma25);
      if (ma99Ref.current) ma99Ref.current.innerText = formatVal(mas?.ma99);
    }
  }));

  const labelStyle = { color: UI_COLORS.textSecondary, fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.05em', mb: '1px' };
  const valStyle = { color: UI_COLORS.textPrimary, fontFamily: '"Roboto Mono", monospace', fontSize: '13px', fontWeight: 500 };
  const maLabelStyle = { color: UI_COLORS.textSecondary, fontSize: '11px' };
  const maValStyle = { fontFamily: '"Roboto Mono", monospace', fontSize: '11px', fontWeight: 500 };

  const getFallbackLogo = () => (
    <Box sx={{ 
      width: 24, height: 24, borderRadius: '50%', bgcolor: 'rgba(255,255,255,0.1)', 
      display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '10px', fontWeight: 700, color: UI_COLORS.textSecondary 
    }}>
      {symbol.substring(0, 1)}
    </Box>
  );

  return (
    <Box sx={{
      display: 'flex', flexDirection: 'column', gap: 1, background: UI_COLORS.overlayBg,
      backdropFilter: 'blur(8px)', p: '10px 14px', borderRadius: 1.5,
      border: '1px solid rgba(71, 77, 87, 0.3)', boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
      minWidth: 'fit-content', pointerEvents: 'none',
    }}>
      <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.2 }}>
          {metadata?.logo && !imgError ? (
            <Box component="img" src={metadata.logo} onError={() => setImgError(true)} sx={{ width: 24, height: 24, borderRadius: '50%', objectFit: 'contain' }} />
          ) : getFallbackLogo()}
          <Box sx={{ display: 'flex', flexDirection: 'column' }}>
            <Typography sx={{ fontWeight: 600, fontSize: '14px', color: UI_COLORS.textPrimary, lineHeight: 1.2 }}>{metadata?.name || symbol}</Typography>
            <Typography sx={{ fontSize: '10px', color: UI_COLORS.textSecondary }}>{symbol}</Typography>
          </Box>
        </Box>

        <Box sx={{ display: 'flex', gap: 1.5 }}>
          <Box sx={{ display: 'flex', flexDirection: 'column' }}><Typography sx={labelStyle}>Open</Typography><Typography ref={openRef} sx={valStyle}>-</Typography></Box>
          <Box sx={{ display: 'flex', flexDirection: 'column' }}><Typography sx={labelStyle}>High</Typography><Typography ref={highRef} sx={valStyle}>-</Typography></Box>
          <Box sx={{ display: 'flex', flexDirection: 'column' }}><Typography sx={labelStyle}>Low</Typography><Typography ref={lowRef} sx={valStyle}>-</Typography></Box>
          <Box sx={{ display: 'flex', flexDirection: 'column' }}><Typography sx={labelStyle}>Close</Typography><Typography ref={closeRef} sx={valStyle}>-</Typography></Box>
          <Box sx={{ display: 'flex', flexDirection: 'column' }}><Typography sx={labelStyle}>Vol</Typography><Typography ref={volRef} sx={valStyle}>-</Typography></Box>
        </Box>
      </Box>

      <Box sx={{ display: 'flex', gap: 1.5, pt: 0.5, borderTop: `1px solid rgba(71, 77, 87, 0.2)` }}>
        <Box sx={{ display: 'flex', gap: 0.5 }}><Typography sx={maLabelStyle}>MA7:</Typography><Typography ref={ma7Ref} sx={{ ...maValStyle, color: UI_COLORS.ma7 }}>-</Typography></Box>
        <Box sx={{ display: 'flex', gap: 0.5 }}><Typography sx={maLabelStyle}>MA25:</Typography><Typography ref={ma25Ref} sx={{ ...maValStyle, color: UI_COLORS.ma25 }}>-</Typography></Box>
        <Box sx={{ display: 'flex', gap: 0.5 }}><Typography sx={maLabelStyle}>MA99:</Typography><Typography ref={ma99Ref} sx={{ ...maValStyle, color: UI_COLORS.ma99 }}>-</Typography></Box>
      </Box>
    </Box>
  );
});

export default StockLegend;
