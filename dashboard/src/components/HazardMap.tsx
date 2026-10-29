"use client";

import React, { useState, useEffect, useRef } from "react";
import { 
  Layers, 
  AlertTriangle, 
  Droplets, 
  Sun, 
  Activity, 
  Users, 
  Compass, 
  MapPin, 
  Maximize2, 
  Info,
  Waves,
  ShieldAlert,
  Globe2,
  Eye,
  RotateCcw,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  FastForward,
  Repeat,
  Clock,
  Sparkles,
  Radio
} from "lucide-react";

export interface SubBasinFeature {
  hybas_id: number;
  name: string;
  region: string;
  river_system: string;
  latLngs: [number, number][];
  center: [number, number];
  flood_prob: number;
  flood_level: string;
  spei: number;
  drought_level: string;
  uncertainty_sigma: number;
  rain_7d: number;
  twi: number;
  pop_density: number;
  infra_count: number;
  action: string;
}

export const NATIONAL_OVERVIEW: SubBasinFeature = {
  hybas_id: 100000,
  name: "All-India Continental Overview",
  region: "National (12 Major River Basins)",
  river_system: "National River Network (Ganga, Brahmaputra, Godavari, Indus, Krishna)",
  center: [22.5, 80.5],
  latLngs: [],
  flood_prob: 0.62,
  flood_level: "HIGH",
  spei: -0.4,
  drought_level: "NORMAL",
  uncertainty_sigma: 0.13,
  rain_7d: 88.0,
  twi: 14.5,
  pop_density: 468,
  infra_count: 630,
  action: "National multi-catchment surveillance active across all 12 major Indian river basins."
};

export const ALL_INDIA_BASINS: SubBasinFeature[] = [
  NATIONAL_OVERVIEW,
  {
    hybas_id: 101001,
    name: "Upper & Middle Ganga Basin",
    region: "Northern India",
    river_system: "Ganga River System",
    center: [26.8, 81.5],
    latLngs: [
      [30.5, 78.0], [31.0, 79.2], [30.0, 80.8], [27.5, 84.5], 
      [26.8, 87.2], [25.5, 88.2], [24.0, 88.5], [23.8, 87.0], 
      [24.5, 84.0], [25.5, 80.5], [27.2, 78.0], [29.2, 77.5]
    ],
    flood_prob: 0.72,
    flood_level: "CRITICAL",
    spei: 0.4,
    drought_level: "NORMAL",
    uncertainty_sigma: 0.12,
    rain_7d: 135.0,
    twi: 16.8,
    pop_density: 890,
    infra_count: 142,
    action: "Critical flood warning: Stage NDRF teams in low-lying riparian corridors."
  },
  {
    hybas_id: 102001,
    name: "Brahmaputra Valley & Delta",
    region: "Northeastern India",
    river_system: "Brahmaputra System",
    center: [26.5, 93.0],
    latLngs: [
      [28.2, 93.5], [28.8, 95.5], [28.0, 96.5], [26.8, 95.2], 
      [26.0, 92.8], [25.4, 90.0], [25.8, 89.8], [26.8, 91.5], [27.5, 92.5]
    ],
    flood_prob: 0.89,
    flood_level: "CRITICAL",
    spei: 0.8,
    drought_level: "NORMAL",
    uncertainty_sigma: 0.11,
    rain_7d: 195.0,
    twi: 18.2,
    pop_density: 460,
    infra_count: 58,
    action: "Emergency flood alert: Active river embankment overflow in Kaziranga & Majuli reaches."
  },
  {
    hybas_id: 103001,
    name: "Indus & Himalayan Headwaters",
    region: "Northern / Trans-Himalaya",
    river_system: "Indus River System",
    center: [33.2, 76.0],
    latLngs: [
      [35.5, 76.5], [34.5, 78.5], [32.8, 77.8], [31.5, 76.5], 
      [31.0, 74.5], [32.5, 74.0], [34.0, 74.2], [35.2, 75.5]
    ],
    flood_prob: 0.34,
    flood_level: "MODERATE",
    spei: -0.8,
    drought_level: "MILD",
    uncertainty_sigma: 0.15,
    rain_7d: 52.0,
    twi: 11.5,
    pop_density: 160,
    infra_count: 32,
    action: "Glacial melt & localized flash flood advisory in mountain river gorges."
  },
  {
    hybas_id: 104001,
    name: "Thar & Luni Arid Basin",
    region: "Western Arid Region",
    river_system: "Internal Drainage (Luni)",
    center: [26.5, 72.5],
    latLngs: [
      [28.5, 72.0], [28.0, 75.0], [26.5, 74.8], [24.8, 73.0], 
      [24.2, 71.0], [26.0, 70.0], [27.5, 70.5]
    ],
    flood_prob: 0.08,
    flood_level: "LOW",
    spei: -2.3,
    drought_level: "EXTREME",
    uncertainty_sigma: 0.14,
    rain_7d: 4.0,
    twi: 7.2,
    pop_density: 125,
    infra_count: 18,
    action: "Critical agricultural drought: Activate groundwater emergency reserves & cattle fodder camps."
  },
  {
    hybas_id: 105001,
    name: "Sabarmati & Mahi Basin",
    region: "Western India",
    river_system: "Sabarmati / Mahi",
    center: [22.8, 72.8],
    latLngs: [
      [24.5, 72.5], [24.0, 74.0], [22.8, 74.2], [21.8, 73.0], 
      [22.2, 71.8], [23.2, 71.2]
    ],
    flood_prob: 0.28,
    flood_level: "MODERATE",
    spei: -1.4,
    drought_level: "MODERATE",
    uncertainty_sigma: 0.13,
    rain_7d: 38.0,
    twi: 12.0,
    pop_density: 410,
    infra_count: 64,
    action: "Monitor reservoir storage levels and agricultural irrigation schedules."
  },
  {
    hybas_id: 106001,
    name: "Narmada & Tapi Valley",
    region: "Central India",
    river_system: "Narmada / Tapi Rift Valley",
    center: [22.0, 76.8],
    latLngs: [
      [23.2, 78.0], [23.0, 81.5], [21.8, 81.0], [21.2, 77.0], 
      [21.0, 73.0], [21.8, 72.6], [22.2, 75.0]
    ],
    flood_prob: 0.46,
    flood_level: "HIGH",
    spei: -0.6,
    drought_level: "MILD",
    uncertainty_sigma: 0.13,
    rain_7d: 82.0,
    twi: 14.4,
    pop_density: 310,
    infra_count: 44,
    action: "Controlled dam sluice discharge alert along Sardar Sarovar downstream reaches."
  },
  {
    hybas_id: 107001,
    name: "Mahanadi Catchment & Delta",
    region: "Eastern India",
    river_system: "Mahanadi Basin",
    center: [21.2, 84.0],
    latLngs: [
      [23.5, 84.0], [22.5, 86.5], [20.5, 86.8], [19.5, 84.5], 
      [20.0, 81.5], [21.8, 81.2]
    ],
    flood_prob: 0.65,
    flood_level: "HIGH",
    spei: 0.2,
    drought_level: "NORMAL",
    uncertainty_sigma: 0.13,
    rain_7d: 110.0,
    twi: 16.1,
    pop_density: 380,
    infra_count: 52,
    action: "Coastal storm runoff warning: Restrict fishing operations in Hirakud tail-waters."
  },
  {
    hybas_id: 406008,
    name: "Godavari Basin & Wardha Plain",
    region: "Deccan Plateau",
    river_system: "Godavari River System",
    center: [19.2, 78.5],
    latLngs: [
      [21.0, 74.5], [21.5, 77.0], [21.2, 80.0], [19.8, 81.5], 
      [17.5, 82.2], [16.5, 81.5], [17.5, 78.0], [18.5, 75.0], [19.8, 73.8]
    ],
    flood_prob: 0.88,
    flood_level: "CRITICAL",
    spei: -0.2,
    drought_level: "NORMAL",
    uncertainty_sigma: 0.14,
    rain_7d: 145.0,
    twi: 17.5,
    pop_density: 720,
    infra_count: 68,
    action: "Immediate evacuation of Wardha & Pranhita floodplains; prepare emergency shelters."
  },
  {
    hybas_id: 108001,
    name: "Krishna Basin Catchment",
    region: "South-Central India",
    river_system: "Krishna River System",
    center: [16.5, 77.0],
    latLngs: [
      [18.8, 75.0], [18.2, 78.5], [17.0, 80.8], [15.8, 80.5], 
      [14.5, 78.0], [13.8, 75.5], [15.5, 74.0], [17.2, 74.0]
    ],
    flood_prob: 0.42,
    flood_level: "MODERATE",
    spei: -1.1,
    drought_level: "MODERATE",
    uncertainty_sigma: 0.13,
    rain_7d: 68.0,
    twi: 13.9,
    pop_density: 460,
    infra_count: 62,
    action: "Regulate inflow into Nagarjuna Sagar; stage river bank patrol."
  },
  {
    hybas_id: 109001,
    name: "Cauvery Basin & Southern Plains",
    region: "Peninsular South",
    river_system: "Cauvery System",
    center: [11.8, 77.5],
    latLngs: [
      [13.5, 77.0], [12.8, 79.0], [11.5, 79.8], [10.5, 78.8], 
      [11.0, 76.5], [12.2, 75.5], [13.2, 76.0]
    ],
    flood_prob: 0.38,
    flood_level: "MODERATE",
    spei: -1.7,
    drought_level: "SEVERE",
    uncertainty_sigma: 0.14,
    rain_7d: 42.0,
    twi: 12.8,
    pop_density: 680,
    infra_count: 76,
    action: "Agricultural drought watch: Enforce rotational canal irrigation schedules."
  },
  {
    hybas_id: 110001,
    name: "Western Ghats Coastal Belt",
    region: "Southwestern Coast",
    river_system: "West-Flowing Rivers",
    center: [14.0, 74.8],
    latLngs: [
      [19.5, 73.0], [18.0, 73.5], [15.0, 74.0], [12.0, 75.0], 
      [9.0, 76.5], [8.2, 77.2], [8.5, 76.8], [13.0, 74.5], [16.0, 73.2]
    ],
    flood_prob: 0.76,
    flood_level: "CRITICAL",
    spei: 1.1,
    drought_level: "NORMAL",
    uncertainty_sigma: 0.12,
    rain_7d: 185.0,
    twi: 17.9,
    pop_density: 820,
    infra_count: 95,
    action: "Flash flood & landslide red alert in steep orographic Western Ghats terrain."
  },
  {
    hybas_id: 111001,
    name: "Southern Peninsula Tip & Kanyakumari",
    region: "Far South",
    river_system: "Vaigai / Tamirabarani",
    center: [9.2, 78.0],
    latLngs: [
      [10.5, 78.0], [9.5, 79.2], [8.5, 78.2], [8.1, 77.5], [9.0, 77.2], [10.0, 77.5]
    ],
    flood_prob: 0.19,
    flood_level: "LOW",
    spei: -0.5,
    drought_level: "NORMAL",
    uncertainty_sigma: 0.13,
    rain_7d: 28.0,
    twi: 11.0,
    pop_density: 540,
    infra_count: 24,
    action: "Normal monitoring conditions across southern maritime drainage catchments."
  }
];

export const PILOT_BASINS = ALL_INDIA_BASINS;

// Real River Trajectories
const MAJOR_RIVERS: { name: string; path: [number, number][] }[] = [
  {
    name: "Ganga River",
    path: [
      [30.98, 78.93], [30.14, 78.29], [29.94, 78.16], [26.44, 80.33], 
      [25.43, 81.84], [25.31, 83.00], [25.59, 85.13], [24.81, 87.92], 
      [22.82, 88.38], [21.80, 88.20]
    ]
  },
  {
    name: "Brahmaputra River",
    path: [
      [28.80, 95.10], [28.06, 95.33], [27.47, 94.91], 
      [26.18, 91.75], [26.02, 89.97], [25.10, 89.70]
    ]
  },
  {
    name: "Godavari River",
    path: [
      [19.93, 73.53], [19.87, 75.34], [19.15, 77.31], 
      [18.78, 78.91], [17.67, 80.89], [17.00, 81.78], [16.71, 82.26]
    ]
  },
  {
    name: "Krishna River",
    path: [
      [17.92, 73.66], [16.85, 74.58], [16.51, 77.32], 
      [16.18, 79.91], [16.50, 80.64], [15.78, 80.91]
    ]
  },
  {
    name: "Narmada River",
    path: [
      [22.67, 81.75], [23.16, 79.95], [22.75, 77.72], 
      [22.18, 75.30], [21.83, 73.74], [21.62, 72.63]
    ]
  },
  {
    name: "Cauvery River",
    path: [
      [12.38, 75.49], [12.42, 76.69], [11.80, 77.80], 
      [10.83, 78.68], [11.13, 79.85]
    ]
  }
];

export type MapLayer = "flood" | "drought" | "uncertainty" | "environment" | "exposure";
export type BasemapType = "dark" | "satellite" | "topo" | "osm";

export interface TimeStepData {
  leadHours: number;
  label: string;
  subLabel: string;
  stageName: string;
  avgVelocity: string;
  description: string;
  estInundationKm2: number;
  criticalCount: number;
  activeSurgeReach: string;
}

export const FLOOD_TIME_STEPS: TimeStepData[] = [
  {
    leadHours: 6,
    label: "+6h",
    subLabel: "T+6 Hours",
    stageName: "Initial Headwater Surge & Orographic Runoff",
    avgVelocity: "5.8 km/h",
    description: "Rapid mountain torrent runoff and surface saturation in steep upper catchments (Western Ghats, Indus).",
    estInundationKm2: 3200,
    criticalCount: 2,
    activeSurgeReach: "Western Ghats Escarpments & Upper Himalayan Gorges"
  },
  {
    leadHours: 12,
    label: "+12h",
    subLabel: "T+12 Hours",
    stageName: "Tributary Channel Influx & Confluence Swell",
    avgVelocity: "5.1 km/h",
    description: "Runoff from sub-tributaries converges into primary river channels; water levels rise rapidly toward warning mark.",
    estInundationKm2: 5800,
    criticalCount: 3,
    activeSurgeReach: "Pranhita Confluence & Upper Ganga Tributaries"
  },
  {
    leadHours: 24,
    label: "+24h",
    subLabel: "T+1 Day",
    stageName: "Mainstem Bankfull Stage & Embankment Stress",
    avgVelocity: "4.4 km/h",
    description: "Flood wave crest builds along middle river corridors; low-lying riparian pasturelands begin inundating.",
    estInundationKm2: 9400,
    criticalCount: 4,
    activeSurgeReach: "Wardha Floodplain, Kaziranga Reaches & Middle Ganga"
  },
  {
    leadHours: 48,
    label: "+48h",
    subLabel: "T+2 Days",
    stageName: "Peak Continental Inundation & Riparian Crest",
    avgVelocity: "3.6 km/h",
    description: "Maximum flood wave amplitude in major alluvial plains (Wardha, Brahmaputra, Ganga). Highest emergency alert level.",
    estInundationKm2: 14200,
    criticalCount: 5,
    activeSurgeReach: "Brahmaputra Valley, Wardha Floodplain & Central Alluvial Plain"
  },
  {
    leadHours: 72,
    label: "+72h",
    subLabel: "T+3 Days",
    stageName: "Deltaic Backwater Accumulation & Estuarine Surge",
    avgVelocity: "2.8 km/h",
    description: "Upper basins draining and receding. Peak discharge arrives at downstream deltas and tidal estuaries with high backwater resistance.",
    estInundationKm2: 11800,
    criticalCount: 3,
    activeSurgeReach: "Lower Godavari Delta Channel, Sundarbans & Mahanadi Outfall"
  }
];

export const DROUGHT_TIME_STEPS = [
  { step: 1, label: "1 Mo", subLabel: "30-Day Deficit", stageName: "Meteorological Drought", description: "Topsoil moisture deficit in rainfed agricultural corridors." },
  { step: 2, label: "2 Mo", subLabel: "60-Day Deficit", stageName: "Agricultural Drought", description: "Root-zone soil dryout and vegetation stress (MODIS NDVI anomaly)." },
  { step: 3, label: "3 Mo", subLabel: "90-Day Deficit", stageName: "Hydrological Drought", description: "Reservoir storage drawdowns and stream baseflow reduction." },
  { step: 6, label: "6 Mo", subLabel: "180-Day Deficit", stageName: "Socio-Economic Drought", description: "Deep groundwater aquifer depletion & municipal emergency water rationing." }
];

export function computeBasinFloodAtTime(basin: SubBasinFeature, leadHours: number) {
  const isHeadwater = [103001, 110001, 406001, 406004, 406007].includes(basin.hybas_id);
  const isDelta = [406009, 406010, 107001, 111001].includes(basin.hybas_id);

  let multiplier = 1.0;
  let waveState: "RISING" | "PEAK" | "RECEDING" | "NOMINAL" = "NOMINAL";

  if (isHeadwater) {
    if (leadHours <= 6) {
      multiplier = 1.25;
      waveState = "PEAK";
    } else if (leadHours <= 12) {
      multiplier = 1.15;
      waveState = "PEAK";
    } else if (leadHours <= 24) {
      multiplier = 0.95;
      waveState = "RECEDING";
    } else if (leadHours <= 48) {
      multiplier = 0.70;
      waveState = "RECEDING";
    } else {
      multiplier = 0.50;
      waveState = "NOMINAL";
    }
  } else if (isDelta) {
    if (leadHours <= 6) {
      multiplier = 0.45;
      waveState = "NOMINAL";
    } else if (leadHours <= 12) {
      multiplier = 0.60;
      waveState = "RISING";
    } else if (leadHours <= 24) {
      multiplier = 0.85;
      waveState = "RISING";
    } else if (leadHours <= 48) {
      multiplier = 1.10;
      waveState = "RISING";
    } else {
      multiplier = 1.35;
      waveState = "PEAK";
    }
  } else {
    // Middle reach / Floodplain
    if (leadHours <= 6) {
      multiplier = 0.65;
      waveState = "RISING";
    } else if (leadHours <= 12) {
      multiplier = 0.80;
      waveState = "RISING";
    } else if (leadHours <= 24) {
      multiplier = 0.95;
      waveState = "RISING";
    } else if (leadHours <= 48) {
      multiplier = 1.0;
      waveState = "PEAK";
    } else {
      multiplier = 0.85;
      waveState = "RECEDING";
    }
  }

  const dynamicProb = Math.min(0.98, Math.max(0.04, basin.flood_prob * multiplier));
  const dynamicLevel = dynamicProb >= 0.70 ? "CRITICAL" : (dynamicProb >= 0.45 ? "HIGH" : (dynamicProb >= 0.20 ? "MODERATE" : "LOW"));

  return {
    prob: dynamicProb,
    level: dynamicLevel,
    state: waveState,
    inundationAreaEstKm2: Math.round(dynamicProb * (basin.pop_density * 4.2))
  };
}

export function computeBasinDroughtAtTime(basin: SubBasinFeature, stepMonths: number) {
  const delta = (stepMonths - 1) * -0.25;
  const dynamicSpei = Math.max(-3.5, Math.min(1.5, basin.spei + delta));
  const dynamicLevel = dynamicSpei <= -2.0 ? "EXTREME" : (dynamicSpei <= -1.5 ? "SEVERE" : (dynamicSpei <= -1.0 ? "MODERATE" : "NORMAL"));
  return {
    spei: dynamicSpei,
    level: dynamicLevel
  };
}

interface HazardMapProps {
  selectedBasinId: number;
  onSelectBasin: (basin: SubBasinFeature) => void;
}

export const HazardMap: React.FC<HazardMapProps> = ({ selectedBasinId, onSelectBasin }) => {
  const [activeLayer, setActiveLayer] = useState<MapLayer>("flood");
  const [basemap, setBasemap] = useState<BasemapType>("dark");
  const [isClient, setIsClient] = useState<boolean>(false);
  const [timeStepIndex, setTimeStepIndex] = useState<number>(3); // Default +48h
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const [isLooping, setIsLooping] = useState<boolean>(true);

  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const tileLayerRef = useRef<any>(null);
  const polygonLayersRef = useRef<{ [key: number]: any }>({});
  const riverLayersRef = useRef<any[]>([]);
  const waveMarkersRef = useRef<any[]>([]);

  // Doppler Rain Radar Overlay State
  const [showRadar, setShowRadar] = useState<boolean>(false);
  const [radarOpacity, setRadarOpacity] = useState<number>(0.75);
  const [radarFrames, setRadarFrames] = useState<{ time: number; path: string }[]>([]);
  const [radarHost, setRadarHost] = useState<string>("https://tilecache.rainviewer.com");
  const [radarFrameIdx, setRadarFrameIdx] = useState<number>(0);
  const [isRadarPlaying, setIsRadarPlaying] = useState<boolean>(false);
  const radarLayerRef = useRef<any>(null);

  const selectedBasin = ALL_INDIA_BASINS.find((b) => b.hybas_id === selectedBasinId) || ALL_INDIA_BASINS[0];

  useEffect(() => {
    setIsClient(true);
  }, []);

  // Fetch Doppler Radar Frames on Mount
  useEffect(() => {
    if (!isClient) return;
    const fetchRadarFrames = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/v1/weather/radar-frames");
        if (res.ok) {
          const data = await res.json();
          const host = data.host || "https://tilecache.rainviewer.com";
          const past = data.radar?.past || [];
          setRadarHost(host);
          setRadarFrames(past);
          if (past.length > 0) {
            setRadarFrameIdx(past.length - 1);
          }
        }
      } catch (err) {
        console.warn("Could not load radar frames from backend, trying direct:", err);
        try {
          const res = await fetch("https://api.rainviewer.com/public/weather-maps.json");
          const data = await res.json();
          const host = data.host || "https://tilecache.rainviewer.com";
          const past = data.radar?.past || [];
          setRadarHost(host);
          setRadarFrames(past);
          if (past.length > 0) {
            setRadarFrameIdx(past.length - 1);
          }
        } catch (e) {
          console.error("Direct radar fetch error:", e);
        }
      }
    };
    fetchRadarFrames();
  }, [isClient]);

  // Update Doppler Radar Tile Layer on Leaflet
  useEffect(() => {
    if (!mapInstanceRef.current || !isClient) return;
    const L = require("leaflet");
    const map = mapInstanceRef.current;

    // Remove existing radar layer if present
    if (radarLayerRef.current) {
      map.removeLayer(radarLayerRef.current);
      radarLayerRef.current = null;
    }

    if (!showRadar || radarFrames.length === 0) return;

    const frame = radarFrames[radarFrameIdx] || radarFrames[radarFrames.length - 1];
    if (!frame) return;

    const radarTileUrl = `${radarHost}${frame.path}/256/{z}/{x}/{y}/2/1_1.png`;

    const layer = L.tileLayer(radarTileUrl, {
      opacity: radarOpacity,
      zIndex: 250,
      maxZoom: 18,
      attribution: 'Doppler Radar &copy; <a href="https://www.rainviewer.com" target="_blank">RainViewer</a>'
    }).addTo(map);

    radarLayerRef.current = layer;

    return () => {
      if (radarLayerRef.current && map) {
        map.removeLayer(radarLayerRef.current);
        radarLayerRef.current = null;
      }
    };
  }, [showRadar, radarFrameIdx, radarOpacity, radarFrames, radarHost, isClient]);

  // Radar Animation Loop Timer
  useEffect(() => {
    if (!isRadarPlaying || radarFrames.length === 0) return;

    const interval = setInterval(() => {
      setRadarFrameIdx((prev) => (prev + 1) % radarFrames.length);
    }, 900);

    return () => clearInterval(interval);
  }, [isRadarPlaying, radarFrames]);

  // Time-lapse playback timer
  useEffect(() => {
    if (!isPlaying) return;

    const intervalMs = Math.round(1800 / playbackSpeed);
    const timer = setInterval(() => {
      setTimeStepIndex((prev) => {
        const maxSteps = activeLayer === "flood" ? FLOOD_TIME_STEPS.length : DROUGHT_TIME_STEPS.length;
        if (prev + 1 >= maxSteps) {
          if (!isLooping) {
            setIsPlaying(false);
            return prev;
          }
          return 0;
        }
        return prev + 1;
      });
    }, intervalMs);

    return () => clearInterval(timer);
  }, [isPlaying, playbackSpeed, isLooping, activeLayer]);

  const getFillColor = (basin: SubBasinFeature) => {
    switch (activeLayer) {
      case "flood": {
        const curLead = FLOOD_TIME_STEPS[timeStepIndex]?.leadHours || 48;
        const dyn = computeBasinFloodAtTime(basin, curLead);
        if (dyn.level === "CRITICAL") return "#ef4444";
        if (dyn.level === "HIGH") return "#f97316";
        if (dyn.level === "MODERATE") return "#eab308";
        return "#22c55e";
      }

      case "drought": {
        const curMonths = DROUGHT_TIME_STEPS[timeStepIndex % DROUGHT_TIME_STEPS.length]?.step || 1;
        const dyn = computeBasinDroughtAtTime(basin, curMonths);
        if (dyn.spei <= -2.0) return "#7f1d1d";
        if (dyn.spei <= -1.5) return "#dc2626";
        if (dyn.spei <= -1.0) return "#ea580c";
        if (dyn.spei <= -0.5) return "#d97706";
        return "#10b981";
      }

      case "uncertainty":
        if (basin.uncertainty_sigma >= 0.14) return "#a855f7";
        if (basin.uncertainty_sigma >= 0.12) return "#c084fc";
        return "#38bdf8";

      case "environment":
        if (basin.rain_7d >= 120) return "#2563eb";
        if (basin.rain_7d >= 60) return "#0ea5e9";
        return "#06b6d4";

      case "exposure":
        if (basin.pop_density >= 700) return "#f43f5e";
        if (basin.pop_density >= 400) return "#fb923c";
        return "#facc15";

      default:
        return "#38bdf8";
    }
  };

  // Basemap Tile URLs (100% Free, Keyless, No Watermarks)
  const getTileUrl = (type: BasemapType) => {
    switch (type) {
      case "dark":
        // Esri World Dark Gray Canvas: 100% free, no key, no watermark
        return "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}";
      case "satellite":
        // Esri World Satellite Imagery: 100% free, no key, no watermark
        return "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
      case "topo":
        // Esri World Topo Map: 100% free, no key, no watermark
        return "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}";
      case "osm":
      default:
        // OpenStreetMap Standard: 100% free, no key
        return "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
    }
  };

  const getTileAttribution = (type: BasemapType) => {
    switch (type) {
      case "dark":
        return '&copy; <a href="https://www.esri.com/">Esri</a> &copy; OpenStreetMap';
      case "satellite":
        return '&copy; <a href="https://www.esri.com/">Esri</a>, Maxar, Earthstar Geographics';
      case "topo":
        return '&copy; <a href="https://www.esri.com/">Esri</a>, USGS';
      case "osm":
      default:
        return '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
    }
  };

  // Initialize Leaflet Map
  useEffect(() => {
    if (!isClient || !mapContainerRef.current) return;

    let L: any;
    try {
      L = require("leaflet");
    } catch {
      return;
    }

    if (!mapInstanceRef.current) {
      const map = L.map(mapContainerRef.current, {
        center: [22.5, 80.5],
        zoom: 5,
        minZoom: 4,
        maxZoom: 14,
        zoomControl: true,
      });

      mapInstanceRef.current = map;

      // Add Tile Layer
      tileLayerRef.current = L.tileLayer(getTileUrl(basemap), {
        attribution: getTileAttribution(basemap),
        maxZoom: 18,
      }).addTo(map);

      // Add River Trajectories
      MAJOR_RIVERS.forEach((river) => {
        const polyline = L.polyline(river.path, {
          color: "#38bdf8",
          weight: 3.5,
          opacity: 0.85,
          dashArray: "6, 6",
          lineCap: "round",
        }).addTo(map);

        polyline.bindTooltip(`<strong>${river.name}</strong><br/>Arterial Hydrologic Drainage`, {
          sticky: true,
          className: "leaflet-custom-tooltip",
        });

        riverLayersRef.current.push(polyline);
      });
    }

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, [isClient]);

  // Update Basemap Tiles
  useEffect(() => {
    if (!mapInstanceRef.current || !tileLayerRef.current) return;
    const L = require("leaflet");

    mapInstanceRef.current.removeLayer(tileLayerRef.current);
    tileLayerRef.current = L.tileLayer(getTileUrl(basemap), {
      attribution: getTileAttribution(basemap),
      maxZoom: 18,
    }).addTo(mapInstanceRef.current);
  }, [basemap]);

  // Update River Wave Crest Markers when timeStepIndex or activeLayer changes
  useEffect(() => {
    if (!mapInstanceRef.current || !isClient) return;
    let L: any;
    try {
      L = require("leaflet");
    } catch {
      return;
    }
    const map = mapInstanceRef.current;

    // Remove previous wave markers
    waveMarkersRef.current.forEach((marker) => {
      map.removeLayer(marker);
    });
    waveMarkersRef.current = [];

    if (activeLayer === "flood") {
      const curStep = FLOOD_TIME_STEPS[timeStepIndex];
      const stepFraction = timeStepIndex / (FLOOD_TIME_STEPS.length - 1);

      MAJOR_RIVERS.forEach((river) => {
        const pathLen = river.path.length;
        if (pathLen === 0) return;
        const ptIdx = Math.min(pathLen - 1, Math.round(stepFraction * (pathLen - 1)));
        const coord = river.path[ptIdx];

        const iconHtml = `
          <div class="flood-wave-marker">
            <div class="pulse-ring"></div>
            <div class="wave-core">🌊</div>
            <div class="wave-tag">${curStep.label} Crest</div>
          </div>
        `;

        const waveMarker = L.marker(coord, {
          icon: L.divIcon({
            className: "leaflet-wave-div-icon",
            html: iconHtml,
            iconSize: [36, 36],
            iconAnchor: [18, 18],
          }),
          zIndexOffset: 1000,
        }).addTo(map);

        waveMarker.bindTooltip(
          `<strong>${river.name} Flood Wave Front</strong><br/>
           Horizon: <span style="color: #38bdf8; font-weight: bold;">${curStep.label} (${curStep.subLabel})</span><br/>
           Wave Stage: <em>${curStep.stageName}</em><br/>
           Front Velocity: <strong>${curStep.avgVelocity}</strong><br/>
           Active Reach: <em>${curStep.activeSurgeReach}</em>`,
          {
            sticky: true,
            className: "leaflet-custom-tooltip",
          }
        );

        waveMarkersRef.current.push(waveMarker);
      });
    }
  }, [timeStepIndex, activeLayer, isClient]);

  // Update Basin Polygons when Layer, Horizon, or Selection changes
  useEffect(() => {
    if (!mapInstanceRef.current) return;
    const L = require("leaflet");
    const map = mapInstanceRef.current;

    // Clear existing polygons
    Object.values(polygonLayersRef.current).forEach((layer) => {
      map.removeLayer(layer);
    });
    polygonLayersRef.current = {};

    const curFloodStep = FLOOD_TIME_STEPS[timeStepIndex];
    const curDroughtStep = DROUGHT_TIME_STEPS[timeStepIndex % DROUGHT_TIME_STEPS.length];

    ALL_INDIA_BASINS.filter((b) => b.latLngs.length > 0).forEach((basin) => {
      const isSelected = basin.hybas_id === selectedBasinId;
      const color = getFillColor(basin);
      const dynFlood = computeBasinFloodAtTime(basin, curFloodStep.leadHours);
      const dynDrought = computeBasinDroughtAtTime(basin, curDroughtStep.step);

      const polygon = L.polygon(basin.latLngs, {
        color: isSelected ? "#38bdf8" : color,
        weight: isSelected ? 3.5 : 1.5,
        opacity: 0.95,
        fillColor: color,
        fillOpacity: isSelected ? 0.65 : 0.40,
      }).addTo(map);

      // Dynamic Interactive Popup reflecting current Horizon
      const popupHtml = `
        <div style="font-family: inherit; min-width: 220px;">
          <div style="font-size: 13px; font-weight: bold; color: #ffffff; margin-bottom: 2px;">
            ${basin.name}
          </div>
          <div style="font-size: 10px; color: #38bdf8; font-weight: 600; text-transform: uppercase; margin-bottom: 6px;">
            ${basin.region} • ${basin.river_system}
          </div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 11px; margin-bottom: 8px;">
            <div style="background: rgba(15, 23, 42, 0.85); padding: 5px 6px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.06);">
              <span style="color: #94a3b8; font-size: 9px; display: block;">${activeLayer === "flood" ? `${curFloodStep.label} Wave Prob` : "30d Drought"}</span>
              <strong style="color: ${dynFlood.level === "CRITICAL" ? "#ef4444" : (dynFlood.level === "HIGH" ? "#f97316" : "#22c55e")}; font-size: 13px;">
                ${(dynFlood.prob * 100).toFixed(0)}% <span style="font-size: 9px; color: #cbd5e1;">(${dynFlood.state})</span>
              </strong>
            </div>
            <div style="background: rgba(15, 23, 42, 0.85); padding: 5px 6px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.06);">
              <span style="color: #94a3b8; font-size: 9px; display: block;">Est. Inundation</span>
              <strong style="color: #38bdf8; font-size: 13px;">
                ${dynFlood.inundationAreaEstKm2.toLocaleString()} km²
              </strong>
            </div>
          </div>
          <div style="font-size: 10px; color: #cbd5e1; margin-bottom: 6px;">
            Horizon: <strong>${curFloodStep.subLabel}</strong> | TWI: <strong>${basin.twi}</strong>
          </div>
          <div style="font-size: 10px; font-style: italic; color: #fde047; margin-bottom: 8px;">
            "${basin.action}"
          </div>
          <button 
            id="popup-btn-${basin.hybas_id}"
            style="width: 100%; padding: 6px; background: #0284c7; color: white; border: none; border-radius: 6px; font-size: 11px; font-weight: bold; cursor: pointer;"
          >
            Inspect This Catchment
          </button>
        </div>
      `;

      polygon.bindPopup(popupHtml);

      polygon.on("popupopen", () => {
        const btn = document.getElementById(`popup-btn-${basin.hybas_id}`);
        if (btn) {
          btn.onclick = () => {
            onSelectBasin(basin);
            polygon.closePopup();
          };
        }
      });

      polygon.on("click", () => {
        onSelectBasin(basin);
      });

      polygonLayersRef.current[basin.hybas_id] = polygon;
    });
  }, [activeLayer, selectedBasinId, isClient, timeStepIndex]);

  // Zoom to basin when selected
  const handleFlyToBasin = (basin: SubBasinFeature) => {
    onSelectBasin(basin);
    if (mapInstanceRef.current) {
      mapInstanceRef.current.flyTo(basin.center, 6.5, {
        duration: 1.2,
      });
      const poly = polygonLayersRef.current[basin.hybas_id];
      if (poly) poly.openPopup();
    }
  };

  const handleResetToIndia = () => {
    onSelectBasin(NATIONAL_OVERVIEW);
    if (mapInstanceRef.current) {
      mapInstanceRef.current.flyTo([22.5, 80.5], 5, { duration: 1.2 });
    }
  };

  return (
    <div className="glass-panel rounded-2xl p-5 shadow-2xl relative overflow-hidden space-y-4 border border-slate-800 bg-slate-900/50">
      {/* Top Header & Layer Controllers */}
      <div className="flex flex-col xl:flex-row items-start xl:items-center justify-between pb-4 border-b border-slate-800 gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="h-3 w-3 rounded-full bg-cyan-400 animate-pulse shadow-lg shadow-cyan-400/50" />
            <h2 className="text-xl font-black text-white tracking-wide flex items-center gap-2">
              National Geographical GIS Hydro-Hazard Map
            </h2>
            <span className="text-[10px] font-extrabold px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800 uppercase font-mono">
              Live Satellite & River Mesh
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Real Geographical Continental Basins | OpenStreetMap / Esri Satellite Tiles | HydroSHEDS Catchment Boundaries
          </p>
        </div>

        {/* 5 Hazard Layer Controls */}
        <div className="flex flex-wrap items-center gap-1.5 p-1 bg-slate-950 rounded-xl border border-slate-800 text-xs font-semibold">
          <button
            onClick={() => setActiveLayer("flood")}
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
              activeLayer === "flood"
                ? "bg-red-500 text-white shadow-md shadow-red-500/30"
                : "text-slate-400 hover:text-white"
            }`}
          >
            <Droplets className="h-3.5 w-3.5" />
            Flood Risk (48h)
          </button>

          <button
            onClick={() => setActiveLayer("drought")}
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
              activeLayer === "drought"
                ? "bg-amber-500 text-slate-950 shadow-md shadow-amber-500/30"
                : "text-slate-400 hover:text-white"
            }`}
          >
            <Sun className="h-3.5 w-3.5" />
            Drought (SPEI)
          </button>

          <button
            onClick={() => setActiveLayer("environment")}
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
              activeLayer === "environment"
                ? "bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/30"
                : "text-slate-400 hover:text-white"
            }`}
          >
            <Waves className="h-3.5 w-3.5" />
            Rainfall / TWI
          </button>

          <button
            onClick={() => setActiveLayer("exposure")}
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
              activeLayer === "exposure"
                ? "bg-rose-500 text-white shadow-md shadow-rose-500/30"
                : "text-slate-400 hover:text-white"
            }`}
          >
            <Users className="h-3.5 w-3.5" />
            Human Exposure
          </button>

          <button
            onClick={() => setActiveLayer("uncertainty")}
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
              activeLayer === "uncertainty"
                ? "bg-purple-500 text-white shadow-md shadow-purple-500/30"
                : "text-slate-400 hover:text-white"
            }`}
          >
            <Activity className="h-3.5 w-3.5" />
            Uncertainty (σ)
          </button>

          {/* Live Doppler Radar Toggle Button */}
          <button
            onClick={() => setShowRadar(!showRadar)}
            id="toggle-doppler-radar-btn"
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 border cursor-pointer ${
              showRadar
                ? "bg-blue-600 text-white border-blue-400 shadow-md shadow-blue-600/40 font-bold"
                : "bg-slate-900 text-slate-300 border-slate-700 hover:text-white"
            }`}
            title="Toggle Live RainViewer Doppler Weather Radar Layer"
          >
            <Radio className="h-3.5 w-3.5" />
            <span>Doppler Radar</span>
            {showRadar && <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse ml-0.5" />}
          </button>
        </div>
      </div>

      {/* Basemap Switcher & Quick Zoom Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-2.5 bg-slate-950/80 rounded-xl border border-slate-800 text-xs">
        {/* Basemap Tile Options */}
        <div className="flex items-center gap-1.5">
          <Globe2 className="h-4 w-4 text-cyan-400" />
          <span className="font-semibold text-slate-300">GIS Basemap:</span>
          <div className="flex items-center gap-1 bg-slate-900 p-0.5 rounded-lg border border-slate-800">
            <button
              onClick={() => setBasemap("dark")}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                basemap === "dark" ? "bg-cyan-500 text-slate-950 font-bold" : "text-slate-400 hover:text-white"
              }`}
            >
              Dark Matter
            </button>
            <button
              onClick={() => setBasemap("satellite")}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                basemap === "satellite" ? "bg-cyan-500 text-slate-950 font-bold" : "text-slate-400 hover:text-white"
              }`}
            >
              Satellite Imagery
            </button>
            <button
              onClick={() => setBasemap("topo")}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                basemap === "topo" ? "bg-cyan-500 text-slate-950 font-bold" : "text-slate-400 hover:text-white"
              }`}
            >
              Terrain Topo
            </button>
            <button
              onClick={() => setBasemap("osm")}
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                basemap === "osm" ? "bg-cyan-500 text-slate-950 font-bold" : "text-slate-400 hover:text-white"
              }`}
            >
              Streets (OSM)
            </button>
          </div>
        </div>

        {/* View Reset Actions */}
        <div className="flex items-center gap-2">
          <button
            onClick={handleResetToIndia}
            className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-xs transition-all border border-slate-700"
          >
            <RotateCcw className="h-3.5 w-3.5 text-cyan-400" />
            Reset to Whole India
          </button>
          <span className="text-slate-500 font-mono text-[11px]">
            Target: <strong className="text-cyan-300">{selectedBasin.name}</strong>
          </span>
        </div>
      </div>

      {/* Live Doppler Weather Radar HUD Controller */}
      {showRadar && (
        <div 
          id="doppler-radar-hud"
          className="p-3.5 rounded-xl bg-slate-950/95 border border-blue-500/40 shadow-xl space-y-2.5 animate-in fade-in duration-200"
        >
          <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2.5">
              <span className="p-2 rounded-lg bg-blue-500/20 text-blue-400 border border-blue-500/40">
                <Radio className="h-4 w-4 animate-pulse" />
              </span>
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-bold text-white text-xs">Doppler Precipitation Radar Scan</span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-blue-950 text-blue-300 border border-blue-800 font-bold">
                    {radarFrames[radarFrameIdx] 
                      ? new Date(radarFrames[radarFrameIdx].time * 1000).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" }) + " IST"
                      : "Live Scan"}
                  </span>
                  <span className="text-[10px] font-mono text-slate-400">
                    Scan {radarFrameIdx + 1} / {Math.max(1, radarFrames.length)}
                  </span>
                </div>
                <p className="text-[11px] text-slate-400">
                  Global RainViewer composite radar mosaic updated every 10 minutes.
                </p>
              </div>
            </div>

            {/* Radar Controls: Play/Pause, Scrubber, Opacity */}
            <div className="flex flex-wrap items-center gap-3">
              {/* Play / Pause Animation Loop */}
              <button
                onClick={() => setIsRadarPlaying(!isRadarPlaying)}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-bold transition-all shadow-sm cursor-pointer ${
                  isRadarPlaying
                    ? "bg-amber-500 text-slate-950 shadow-amber-500/20"
                    : "bg-blue-600 hover:bg-blue-500 text-white shadow-blue-600/20"
                }`}
              >
                {isRadarPlaying ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
                <span>{isRadarPlaying ? "Pause Loop" : "Play Loop"}</span>
              </button>

              {/* Frame Scrubber Slider */}
              <div className="flex items-center gap-2 text-[11px] font-mono text-slate-300">
                <span className="text-slate-500">-2h</span>
                <input
                  type="range"
                  min={0}
                  max={Math.max(0, radarFrames.length - 1)}
                  value={radarFrameIdx}
                  onChange={(e) => {
                    setIsRadarPlaying(false);
                    setRadarFrameIdx(Number(e.target.value));
                  }}
                  className="w-24 sm:w-36 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <span className="text-blue-400 font-bold">Now</span>
              </div>

              {/* Opacity Slider */}
              <div className="flex items-center gap-2 text-[11px] font-mono text-slate-400 border-l border-slate-800 pl-3">
                <span>Opacity:</span>
                <input
                  type="range"
                  min={0.2}
                  max={1.0}
                  step={0.05}
                  value={radarOpacity}
                  onChange={(e) => setRadarOpacity(Number(e.target.value))}
                  className="w-16 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                />
                <span className="text-slate-300 text-[10px] w-7">{Math.round(radarOpacity * 100)}%</span>
              </div>
            </div>
          </div>

          {/* Color Scale Legend */}
          <div className="flex flex-wrap items-center justify-between gap-2 text-[10px] text-slate-400 pt-1.5 border-t border-slate-800/80 font-mono">
            <span className="text-slate-500">Reflectivity / Rain Rate (dBZ):</span>
            <div className="flex flex-wrap items-center gap-2">
              <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-cyan-400" /> Drizzle (10 dBZ)</span>
              <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-blue-500" /> Light Rain (25 dBZ)</span>
              <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-emerald-400" /> Moderate (35 dBZ)</span>
              <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-amber-400" /> Heavy (45 dBZ)</span>
              <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-red-500" /> Cloudburst (55+ dBZ)</span>
            </div>
          </div>
        </div>
      )}

      {/* Real Geographic Leaflet Map Canvas */}
      <div className="relative rounded-2xl overflow-hidden border border-slate-800 shadow-2xl bg-slate-950 min-h-[580px]">
        {/* Leaflet DOM Node */}
        <div 
          ref={mapContainerRef} 
          id="real-gis-map" 
          className="w-full h-[580px] z-10" 
        />

        {/* Floating Interactive Layer Legend */}
        <div className="absolute bottom-4 left-4 z-20 p-3 bg-slate-900/95 backdrop-blur-md rounded-xl border border-slate-800 text-xs shadow-2xl pointer-events-auto">
          <div className="font-bold text-slate-200 mb-1.5 flex items-center gap-1.5 border-b border-slate-800 pb-1">
            <Layers className="h-3.5 w-3.5 text-cyan-400" />
            Active Layer: {activeLayer.toUpperCase()}
          </div>
          {activeLayer === "flood" && (
            <div className="space-y-1 font-medium text-[11px]">
              <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-red-500 shadow-sm shadow-red-500/50" /> Critical (≥ 70%)</div>
              <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-orange-500" /> High (45 - 69%)</div>
              <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-yellow-500" /> Moderate (20 - 44%)</div>
              <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-green-500" /> Low (&lt; 20%)</div>
            </div>
          )}
          {activeLayer === "drought" && (
            <div className="space-y-1 font-medium text-[11px]">
              <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-red-950" /> Extreme Deficit (SPEI ≤ -2.0)</div>
              <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-red-600" /> Severe (SPEI ≤ -1.5)</div>
              <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-orange-500" /> Moderate (SPEI ≤ -1.0)</div>
              <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-emerald-500" /> Normal Hydrologic Balance</div>
            </div>
          )}
          {activeLayer === "environment" && (
            <div className="space-y-1 font-medium text-[11px]">
              <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-blue-600" /> Extreme Rainfall (&gt; 120 mm)</div>
              <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-sky-500" /> Moderate Inundation Potential</div>
            </div>
          )}
          {activeLayer === "exposure" && (
            <div className="space-y-1 font-medium text-[11px]">
              <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-rose-500" /> Dense Megacities (&gt; 700/km²)</div>
              <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-amber-500" /> Agrarian Floodplains</div>
            </div>
          )}
          {activeLayer === "uncertainty" && (
            <div className="space-y-1 font-medium text-[11px]">
              <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-purple-500" /> High Sensor Variance (σ ≥ 0.14)</div>
              <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-sky-500" /> High Model Confidence</div>
            </div>
          )}
        </div>

        {/* Floating Basin Quick Access Card */}
        <div className="absolute top-4 right-4 z-20 max-w-[280px] p-3.5 bg-slate-900/95 backdrop-blur-md rounded-xl border border-slate-800 text-xs shadow-2xl hidden sm:block pointer-events-auto">
          <div className="flex items-center justify-between border-b border-slate-800 pb-1.5 mb-2">
            <span className="text-[10px] font-mono text-cyan-400 font-bold uppercase">Quick Catchment Zoom</span>
            <MapPin className="h-3.5 w-3.5 text-cyan-400" />
          </div>
          <div className="space-y-1 max-h-[160px] overflow-y-auto pr-1">
            {ALL_INDIA_BASINS.map((b) => (
              <button
                key={b.hybas_id}
                onClick={() => handleFlyToBasin(b)}
                className={`w-full text-left px-2 py-1 rounded text-[11px] transition-all flex items-center justify-between ${
                  b.hybas_id === selectedBasinId
                    ? "bg-cyan-500 text-slate-950 font-bold"
                    : "text-slate-300 hover:bg-slate-800 hover:text-white"
                }`}
              >
                <span className="truncate">{b.name}</span>
                <span className="text-[9px] font-mono shrink-0 ml-1">
                  {(b.flood_prob * 100).toFixed(0)}%
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 3. TIME-LAPSE FLOOD WAVE SCRUBBER & TELEMETRY HUD */}
      <div className="p-4 sm:p-5 rounded-2xl bg-slate-950/90 border border-slate-800 shadow-2xl space-y-4">
        {/* Scrubber Header & Playback Controls */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-cyan-950 text-cyan-400 border border-cyan-800/50 shadow-lg shadow-cyan-950/50">
              <Clock className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-extrabold text-white tracking-wide flex items-center gap-2">
                  Spatiotemporal Hazard Propagation Scrubber
                </h3>
                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 flex items-center gap-1.5">
                  <span className="h-1.5 w-1.5 rounded-full bg-cyan-400 animate-ping" />
                  {activeLayer === "flood" ? FLOOD_TIME_STEPS[timeStepIndex].label : DROUGHT_TIME_STEPS[timeStepIndex % DROUGHT_TIME_STEPS.length].label} Horizon
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                {activeLayer === "flood" 
                  ? "Hydrologically routed flood wave attenuation & surge propagation down Indian river corridors" 
                  : "Continuous meteorological-to-groundwater drought deficit progression"}
              </p>
            </div>
          </div>

          {/* Player Button Cluster */}
          <div className="flex items-center gap-2 self-end sm:self-auto">
            {/* Step Back */}
            <button
              onClick={() => {
                const max = activeLayer === "flood" ? FLOOD_TIME_STEPS.length : DROUGHT_TIME_STEPS.length;
                setTimeStepIndex((prev) => (prev - 1 + max) % max);
              }}
              className="p-2 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 hover:text-white transition-all"
              title="Previous Horizon Step"
            >
              <SkipBack className="h-4 w-4" />
            </button>

            {/* Play / Pause Toggle */}
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className={`flex items-center gap-1.5 px-3.5 py-2 rounded-xl font-bold text-xs transition-all shadow-lg ${
                isPlaying
                  ? "bg-amber-500 text-slate-950 shadow-amber-500/30 hover:bg-amber-400"
                  : "bg-cyan-500 text-slate-950 shadow-cyan-500/30 hover:bg-cyan-400"
              }`}
            >
              {isPlaying ? (
                <>
                  <Pause className="h-4 w-4 fill-current" />
                  Pause Wave
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 fill-current" />
                  Simulate Wave
                </>
              )}
            </button>

            {/* Step Forward */}
            <button
              onClick={() => {
                const max = activeLayer === "flood" ? FLOOD_TIME_STEPS.length : DROUGHT_TIME_STEPS.length;
                setTimeStepIndex((prev) => (prev + 1) % max);
              }}
              className="p-2 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 hover:text-white transition-all"
              title="Next Horizon Step"
            >
              <SkipForward className="h-4 w-4" />
            </button>

            {/* Speed Toggle */}
            <button
              onClick={() => setPlaybackSpeed(playbackSpeed === 1 ? 2 : 1)}
              className="px-2.5 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 text-xs font-mono font-bold hover:text-white transition-all"
              title="Toggle Simulation Speed"
            >
              {playbackSpeed}x
            </button>

            {/* Loop Toggle */}
            <button
              onClick={() => setIsLooping(!isLooping)}
              className={`p-2 rounded-lg border transition-all ${
                isLooping
                  ? "bg-cyan-950/80 text-cyan-300 border-cyan-800/60"
                  : "bg-slate-900 text-slate-500 border-slate-800 hover:text-slate-300"
              }`}
              title={isLooping ? "Loop Playback: On" : "Loop Playback: Off"}
            >
              <Repeat className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Interactive Track & Timeline Notch Buttons */}
        <div className="space-y-2.5">
          {/* Timeline Notch Buttons */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
            {activeLayer === "flood"
              ? FLOOD_TIME_STEPS.map((step, idx) => {
                  const isSelected = idx === timeStepIndex;
                  return (
                    <button
                      key={step.leadHours}
                      onClick={() => setTimeStepIndex(idx)}
                      className={`p-2.5 rounded-xl border text-left transition-all ${
                        isSelected
                          ? "bg-gradient-to-br from-cyan-950/90 to-blue-950/90 border-cyan-500/60 shadow-lg shadow-cyan-950/60 ring-1 ring-cyan-500/30"
                          : "bg-slate-900/60 border-slate-800 hover:bg-slate-800/60 text-slate-400 hover:text-white"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className={`text-xs font-black font-mono ${isSelected ? "text-cyan-300" : "text-slate-300"}`}>
                          {step.label}
                        </span>
                        <span className="text-[10px] text-slate-500 font-mono hidden md:inline">
                          {step.subLabel}
                        </span>
                      </div>
                      <p className="text-[10px] text-slate-400 mt-1 truncate">
                        {step.stageName}
                      </p>
                    </button>
                  );
                })
              : DROUGHT_TIME_STEPS.map((step, idx) => {
                  const isSelected = idx === (timeStepIndex % DROUGHT_TIME_STEPS.length);
                  return (
                    <button
                      key={step.step}
                      onClick={() => setTimeStepIndex(idx)}
                      className={`p-2.5 rounded-xl border text-left transition-all ${
                        isSelected
                          ? "bg-gradient-to-br from-amber-950/90 to-orange-950/90 border-amber-500/60 shadow-lg shadow-amber-950/60 ring-1 ring-amber-500/30"
                          : "bg-slate-900/60 border-slate-800 hover:bg-slate-800/60 text-slate-400 hover:text-white"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className={`text-xs font-black font-mono ${isSelected ? "text-amber-300" : "text-slate-300"}`}>
                          {step.label}
                        </span>
                        <span className="text-[10px] text-slate-500 font-mono hidden md:inline">
                          {step.subLabel}
                        </span>
                      </div>
                      <p className="text-[10px] text-slate-400 mt-1 truncate">
                        {step.stageName}
                      </p>
                    </button>
                  );
                })}
          </div>

          {/* Continuous Range Slider */}
          <div className="px-1 pt-1.5">
            <input
              type="range"
              min={0}
              max={activeLayer === "flood" ? FLOOD_TIME_STEPS.length - 1 : DROUGHT_TIME_STEPS.length - 1}
              step={1}
              value={timeStepIndex % (activeLayer === "flood" ? FLOOD_TIME_STEPS.length : DROUGHT_TIME_STEPS.length)}
              onChange={(e) => setTimeStepIndex(Number(e.target.value))}
              className="w-full accent-cyan-400 h-2 bg-slate-900 rounded-lg cursor-pointer"
            />
          </div>
        </div>

        {/* Real-time Wave Dynamics Telemetry HUD Card */}
        {activeLayer === "flood" ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 pt-2 text-xs">
            <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block">Wave Front Velocity</span>
              <strong className="text-cyan-300 font-mono text-sm">{FLOOD_TIME_STEPS[timeStepIndex].avgVelocity}</strong>
              <p className="text-[10px] text-slate-500">Manning-Strickler gravity channel flow</p>
            </div>

            <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block">Active Wave Phase</span>
              <strong className="text-white text-xs truncate block">{FLOOD_TIME_STEPS[timeStepIndex].stageName}</strong>
              <p className="text-[10px] text-slate-500 truncate">{FLOOD_TIME_STEPS[timeStepIndex].activeSurgeReach}</p>
            </div>

            <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block">Critical Basins Active</span>
              <strong className="text-red-400 font-mono text-sm">{FLOOD_TIME_STEPS[timeStepIndex].criticalCount} River Basins</strong>
              <p className="text-[10px] text-slate-500">Calibrated P(Flood) ≥ 70% reached</p>
            </div>

            <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block">Est. Inundation Footprint</span>
              <strong className="text-cyan-300 font-mono text-sm">~{FLOOD_TIME_STEPS[timeStepIndex].estInundationKm2.toLocaleString()} km²</strong>
              <p className="text-[10px] text-slate-500">Continental floodplain coverage</p>
            </div>
          </div>
        ) : (
          <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between text-xs">
            <span className="text-slate-300">
              <strong>Drought Deficit Horizon:</strong> {DROUGHT_TIME_STEPS[timeStepIndex % DROUGHT_TIME_STEPS.length].description}
            </span>
            <span className="font-mono text-amber-400 font-bold">
              {DROUGHT_TIME_STEPS[timeStepIndex % DROUGHT_TIME_STEPS.length].subLabel}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
