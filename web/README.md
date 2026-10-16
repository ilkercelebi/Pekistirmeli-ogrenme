# Arayüz (React + Three.js)

```bash
npm install
npm run build      # web/dist → python serve.py bunu http://localhost:8000 adresinde sunar
npm run dev        # geliştirme: http://localhost:5173 (API için ayrıca: python serve.py)
```

- `src/three/` — 3D robot hücresi. `Robot.tsx` kavrama animasyonunu, API'den gelen olaya (`GraspEvent`) göre
  anahtar karelerle kurar: hedefe git → in → kavra → kaldır → tepsiye bırak / kayıp düş / duvara çarp.
- `src/pages/` — sayfalar. `src/api.ts` — API tipleri ve çağrıları. `src/hooks.ts` — oturum ve animasyon senkronu.
