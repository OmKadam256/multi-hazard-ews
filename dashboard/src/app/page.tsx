"use client";

import React, { useState } from "react";
import { 
  ShieldAlert, 
  Activity, 
  Cpu, 
  Layers, 
  Database, 
  Calendar,
  AlertTriangle, 
  Flame, 
  Droplets, 
  Sun,
  Waves,
  Terminal, 
  ExternalLink, 
  Info,
  Compass,
  MapPin,
  TrendingUp,
  Sliders,
  Sparkles,
  Trophy,
  BookOpen,
  CheckCircle2,
  Globe2,
  Clock,
  Users,
  FileDown,
  FileText
} from "lucide-react";

import { HazardMap, SubBasinFeature, ALL_INDIA_BASINS, NATIONAL_OVERVIEW } from "@/components/HazardMap";
import { ActiveWarningsPanel } from "@/components/ActiveWarningsPanel";
import { AblationScorecard } from "@/components/AblationScorecard";
import { XaiInspector } from "@/components/XaiInspector";
import { RealtimePredictor } from "@/components/RealtimePredictor";
import { LocationDetailsPanel } from "@/components/LocationDetailsPanel";
import { DataQualityPanel } from "@/components/DataQualityPanel";
import { HistoricalTrendsChart } from "@/components/HistoricalTrendsChart";
import { IncidentBriefingModal } from "@/components/IncidentBriefingModal";
import { LiveWeatherForecastWidget } from "@/components/LiveWeatherForecastWidget";

export type NavTab = "overview" | "map" | "simulator" | "historical" | "models" | "datahealth" | "methodology";

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<NavTab>("overview");
  const [selectedBasin, setSelectedBasin] = useState<SubBasinFeature>(NATIONAL_OVERVIEW);
  const [selectedHorizon, setSelectedHorizon] = useState<"48h" | "7d" | "30d">("48h");
  const [isBriefingModalOpen, setIsBriefingModalOpen] = useState<boolean>(false);

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-slate-950">
      
      {/* 1. TOP GLOBAL NAVIGATION BAR */}
      <header className="sticky top-0 z-50 bg-slate-950/90 backdrop-blur-md border-b border-slate-800 px-4 sm:px-6 lg:px-8 py-3.5 shadow-xl">
        <div className="max-w-[1700px] mx-auto flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
          
          {/* Brand Identity */}
          <div className="flex items-center gap-3.5">
            <div className="p-2.5 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl shadow-lg shadow-cyan-500/25 shrink-0">
              <ShieldAlert className="h-6 w-6 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-black tracking-tight text-white flex items-center gap-2">
                  GeoAlert AI
                  <span className="text-[10px] font-extrabold px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800 tracking-wider uppercase">
                    National Observatory
                  </span>
                </h1>
              </div>
              <p className="text-xs text-slate-400 font-medium hidden sm:block">
                Professional Multi-Hazard Early Warning & Geointelligence Platform (Flood + Drought)
              </p>
            </div>
          </div>

          {/* Region & Horizon Quick Pickers */}
          <div className="flex flex-wrap items-center gap-2.5 text-xs">
            {/* Catchment Selector */}
            <div className="flex items-center gap-1.5 bg-slate-900 border border-slate-800 rounded-xl px-2.5 py-1.5">
              <Compass className="h-3.5 w-3.5 text-cyan-400" />
              <span className="text-slate-400">Target:</span>
              <select
                value={selectedBasin.hybas_id}
                onChange={(e) => {
                  const b = ALL_INDIA_BASINS.find((item) => item.hybas_id === Number(e.target.value));
                  if (b) setSelectedBasin(b);
                }}
                className="bg-transparent text-slate-200 font-bold focus:outline-none cursor-pointer"
              >
                {ALL_INDIA_BASINS.map((b) => (
                  <option key={b.hybas_id} value={b.hybas_id} className="bg-slate-900 text-white">
                    {b.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Forecast Horizon */}
            <div className="flex items-center gap-1.5 bg-slate-900 border border-slate-800 rounded-xl px-2.5 py-1.5 font-mono">
              <Calendar className="h-3.5 w-3.5 text-amber-400" />
              <span className="text-slate-400">Horizon:</span>
              <select
                value={selectedHorizon}
                onChange={(e) => setSelectedHorizon(e.target.value as any)}
                className="bg-transparent text-slate-200 font-bold focus:outline-none cursor-pointer"
              >
                <option value="48h" className="bg-slate-900 text-white">48h (Flood Peak)</option>
                <option value="7d" className="bg-slate-900 text-white">7d (Weekly Accumulation)</option>
                <option value="30d" className="bg-slate-900 text-white">30d (SPEI Drought)</option>
              </select>
            </div>

            {/* 1-Click Official Incident Briefing Dossier PDF Export Button */}
            <button
              onClick={() => setIsBriefingModalOpen(true)}
              id="export-dossier-pdf-btn"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold transition-all shadow-md shadow-cyan-500/20 active:scale-95 cursor-pointer"
              title="View and download official CWC / NDRF Incident Action Dossier PDF"
            >
              <FileDown className="h-3.5 w-3.5" />
              <span>Dossier PDF (CWC/NDRF)</span>
            </button>

            {/* Live Pipeline Status Pill */}
            <div className="hidden xl:flex items-center gap-2 px-3 py-1.5 rounded-xl bg-emerald-950/40 border border-emerald-800/60 font-mono text-emerald-300">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>GEE Live Satellite Stream: ONLINE</span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs Bar */}
        <nav className="max-w-[1700px] mx-auto mt-3 pt-2.5 border-t border-slate-800/80 flex items-center gap-1 overflow-x-auto text-xs font-semibold scrollbar-none">
          <button
            onClick={() => setActiveTab("overview")}
            className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-all shrink-0 ${
              activeTab === "overview"
                ? "bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20"
                : "text-slate-400 hover:text-white hover:bg-slate-900"
            }`}
          >
            <Activity className="h-3.5 w-3.5" />
            Overview
          </button>

          <button
            onClick={() => setActiveTab("map")}
            className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-all shrink-0 ${
              activeTab === "map"
                ? "bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20"
                : "text-slate-400 hover:text-white hover:bg-slate-900"
            }`}
          >
            <Layers className="h-3.5 w-3.5" />
            Interactive Map
          </button>

          <button
            onClick={() => setActiveTab("simulator")}
            className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-all shrink-0 ${
              activeTab === "simulator"
                ? "bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20"
                : "text-slate-400 hover:text-white hover:bg-slate-900"
            }`}
          >
            <Sliders className="h-3.5 w-3.5" />
            Forecast Simulator
          </button>

          <button
            onClick={() => setActiveTab("historical")}
            className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-all shrink-0 ${
              activeTab === "historical"
                ? "bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20"
                : "text-slate-400 hover:text-white hover:bg-slate-900"
            }`}
          >
            <TrendingUp className="h-3.5 w-3.5" />
            Historical Analysis
          </button>

          <button
            onClick={() => setActiveTab("models")}
            className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-all shrink-0 ${
              activeTab === "models"
                ? "bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20"
                : "text-slate-400 hover:text-white hover:bg-slate-900"
            }`}
          >
            <Sparkles className="h-3.5 w-3.5" />
            Model Insights & XAI
          </button>

          <button
            onClick={() => setActiveTab("datahealth")}
            className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-all shrink-0 ${
              activeTab === "datahealth"
                ? "bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20"
                : "text-slate-400 hover:text-white hover:bg-slate-900"
            }`}
          >
            <Database className="h-3.5 w-3.5" />
            Data Health & Telemetry
          </button>

          <button
            onClick={() => setActiveTab("methodology")}
            className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-all shrink-0 ${
              activeTab === "methodology"
                ? "bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20"
                : "text-slate-400 hover:text-white hover:bg-slate-900"
            }`}
          >
            <BookOpen className="h-3.5 w-3.5" />
            Methodology & Public Safety
          </button>
        </nav>
      </header>

      {/* 2. MANDATORY OPERATIONAL NOTICE BANNER */}
      <div className="bg-amber-500/10 border-b border-amber-500/30 px-4 sm:px-6 lg:px-8 py-2.5">
        <div className="max-w-[1700px] mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-2.5 text-amber-200">
            <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0" />
            <span>
              <strong>Research Prototype — Decision Support Only.</strong> Model-generated forecasts are experimental research outputs and do not replace official disaster warnings or instructions from local authorities.
            </span>
          </div>

          <div className="flex items-center gap-4 text-slate-400 font-mono text-[11px] shrink-0">
            <span className="text-amber-400">Official Warning Gateways:</span>
            <a
              href="https://mausam.imd.gov.in"
              target="_blank"
              rel="noreferrer"
              className="text-cyan-400 hover:underline flex items-center gap-1"
            >
              IMD Weather <ExternalLink className="h-3 w-3" />
            </a>
            <a
              href="http://cwc.gov.in"
              target="_blank"
              rel="noreferrer"
              className="text-cyan-400 hover:underline flex items-center gap-1"
            >
              CWC Flood Bulletin <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </div>
      </div>

      {/* 3. MAIN DASHBOARD CONTENT AREA */}
      <div className="flex-1 max-w-[1700px] w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-8">

        {/* TAB 1: OVERVIEW */}
        {activeTab === "overview" && (
          <div className="space-y-8">
            {/* Executive KPI Summary Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Flood Outlook */}
              <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold flex items-center gap-1.5">
                    <Droplets className="h-4 w-4 text-red-400" />
                    48-Hour Flood Outlook
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-red-500/20 text-red-300 font-mono">
                    CRITICAL (4 Basins)
                  </span>
                </div>
                <div className="text-2xl font-black text-white font-mono">89% Peak Risk</div>
                <p className="text-[11px] text-slate-400">
                  High alert in <strong>Brahmaputra Valley</strong> and <strong>Wardha Plain</strong>.
                </p>
              </div>

              {/* Drought Outlook */}
              <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold flex items-center gap-1.5">
                    <Sun className="h-4 w-4 text-amber-400" />
                    30-Day Drought (SPEI)
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-300 font-mono">
                    EXTREME DEFICIT
                  </span>
                </div>
                <div className="text-2xl font-black text-amber-400 font-mono">SPEI -2.3</div>
                <p className="text-[11px] text-slate-400">
                  Severe soil deficit in <strong>Thar & Luni Arid Basin</strong>.
                </p>
              </div>

              {/* Telemetry Freshness */}
              <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold flex items-center gap-1.5">
                    <Clock className="h-4 w-4 text-cyan-400" />
                    Telemetry Freshness
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 font-mono">
                    NOMINAL
                  </span>
                </div>
                <div className="text-2xl font-black text-white font-mono">2.5h Ingest Lag</div>
                <p className="text-[11px] text-slate-400">
                  CHIRPS Daily & Sentinel-1 C-SAR validly ingested.
                </p>
              </div>

              {/* Monitored Domain */}
              <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold flex items-center gap-1.5">
                    <Globe2 className="h-4 w-4 text-purple-400" />
                    Continental Coverage
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-500/20 text-purple-300 font-mono">
                    1 km Grid
                  </span>
                </div>
                <div className="text-2xl font-black text-white font-mono">12 River Basins</div>
                <p className="text-[11px] text-slate-400">
                  HydroSHEDS Level-7 Catchments across India.
                </p>
              </div>
            </div>

            {/* Desktop Two-Column Layout: Map (Left) + Prioritized Alerts (Right) */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
              <div className="lg:col-span-8">
                <HazardMap
                  selectedBasinId={selectedBasin.hybas_id}
                  onSelectBasin={(b) => setSelectedBasin(b)}
                />
              </div>
              <div className="lg:col-span-4">
                <ActiveWarningsPanel />
              </div>
            </div>

            {/* Live 7-Day Numerical Weather & Precipitation Forecast (Open-Meteo) */}
            <div>
              <LiveWeatherForecastWidget
                hybasId={selectedBasin.hybas_id}
                basinName={selectedBasin.name}
              />
            </div>

            {/* Contextual Selected Location Assessment */}
            <div>
              <LocationDetailsPanel
                hybasId={selectedBasin.hybas_id}
                basinName={selectedBasin.name}
              />
            </div>
          </div>
        )}

        {/* TAB 2: INTERACTIVE MAP */}
        {activeTab === "map" && (
          <div className="space-y-8">
            <HazardMap
              selectedBasinId={selectedBasin.hybas_id}
              onSelectBasin={(b) => setSelectedBasin(b)}
            />
            <LiveWeatherForecastWidget
              hybasId={selectedBasin.hybas_id}
              basinName={selectedBasin.name}
            />
            <LocationDetailsPanel
              hybasId={selectedBasin.hybas_id}
              basinName={selectedBasin.name}
            />
          </div>
        )}

        {/* TAB 3: FORECAST SIMULATOR */}
        {activeTab === "simulator" && (
          <div className="space-y-8">
            <RealtimePredictor />
          </div>
        )}

        {/* TAB 4: HISTORICAL ANALYSIS */}
        {activeTab === "historical" && (
          <div className="space-y-8">
            <HistoricalTrendsChart basinName={selectedBasin.name} />
          </div>
        )}

        {/* TAB 5: MODEL INSIGHTS & XAI */}
        {activeTab === "models" && (
          <div className="space-y-8">
            <XaiInspector
              basinId={selectedBasin.hybas_id}
              basinName={selectedBasin.name}
            />
            <AblationScorecard />
          </div>
        )}

        {/* TAB 6: DATA HEALTH */}
        {activeTab === "datahealth" && (
          <div className="space-y-8">
            <DataQualityPanel />
          </div>
        )}

        {/* TAB 7: METHODOLOGY & PUBLIC SAFETY */}
        {activeTab === "methodology" && (
          <div className="glass-panel rounded-2xl p-6 sm:p-8 shadow-2xl border border-slate-800 space-y-6 bg-slate-900/60 leading-relaxed text-sm text-slate-300">
            <div className="flex items-center gap-3 border-b border-slate-800 pb-4">
              <BookOpen className="h-6 w-6 text-cyan-400" />
              <div>
                <h2 className="text-xl font-bold text-white">Scientific Methodology & Ethical Safety Governance</h2>
                <p className="text-xs text-slate-400">Architectural principles, zero-leakage protocols, and official authority handoff</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3 p-4 bg-slate-950 rounded-xl border border-slate-800">
                <h3 className="font-bold text-white text-base flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                  1. Zero Data Leakage Enforcement
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Hydrological early warning models must never see observations recorded on or after the forecast issue timestamp $t$. Sentinel-1 SAR and CHIRPS precipitation sequences are strictly windowed to $[t-30d, t)$. Post-event satellite inundation masks are strictly isolated for benchmark ground-truth verification and never supplied as predictive features.
                </p>
              </div>

              <div className="space-y-3 p-4 bg-slate-950 rounded-xl border border-slate-800">
                <h3 className="font-bold text-white text-base flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                  2. HydroSHEDS Spatial Separation
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Pixels are never partitioned randomly across train and test sets. We implement strict <em>Leave-Sub-Basin-Out</em> cross-validation along HydroSHEDS hydrologic catchments. The model is evaluated strictly on basins whose geographic terrain and river networks were never encountered during neural network parameter optimization.
                </p>
              </div>

              <div className="space-y-3 p-4 bg-slate-950 rounded-xl border border-slate-800">
                <h3 className="font-bold text-white text-base flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                  3. Decoupling Hazard from Human Exposure
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Physical hazard probability ($P \in [0, 1]$) is mathematically conditioned solely on physical meteorology, soil moisture, and topographic metrics. Human asset counts, population densities, and infrastructure grids are reported in an explicit exposure context layer to prevent models from artificially inflating hazard probabilities simply because a region is densely populated.
                </p>
              </div>

              <div className="space-y-3 p-4 bg-slate-950 rounded-xl border border-slate-800">
                <h3 className="font-bold text-white text-base flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                  4. Hysteresis Anti-Flapping Engine
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  To eliminate alarm fatigue caused by borderline oscillating probabilities, our operational risk engine enforces a dual-threshold deadband ($\Delta = 0.05$) and requires risk persistence across at least two consecutive timesteps prior to alert escalation or cancellation.
                </p>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-800/40 text-xs text-amber-200 space-y-2">
              <h4 className="font-bold text-amber-300 uppercase tracking-wider text-[11px] flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 text-amber-400" />
                Ethical Directive on Public Safety
              </h4>
              <p>
                This platform is an AI research decision-support tool. It is not an officially certified meteorological warning service. In the event of real-world emergencies, always defer immediately to alerts from the <strong>India Meteorological Department (IMD)</strong>, <strong>Central Water Commission (CWC)</strong>, and the <strong>National Disaster Management Authority (NDMA)</strong>.
              </p>
            </div>
          </div>
        )}

      </div>

      {/* 4. FOOTER */}
      <footer className="border-t border-slate-800 bg-slate-950/80 px-4 sm:px-6 lg:px-8 py-6 text-xs text-slate-500">
        <div className="max-w-[1700px] mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <ShieldAlert className="h-4 w-4 text-cyan-400" />
            <span className="font-bold text-slate-300">GeoAlert AI Geointelligence Platform</span>
            <span>&copy; 2026</span>
            <span className="hidden sm:inline">•</span>
            <span className="hidden sm:inline">Next.js 15 + FastAPI + PyTorch + HydroSHEDS</span>
          </div>

          <div className="flex flex-wrap items-center gap-4 font-mono text-[11px] text-slate-400">
            <span>Projection: EPSG:32644 (UTM-44N)</span>
            <span>•</span>
            <span>Resolution: 1 km</span>
            <span>•</span>
            <span>Dual T4 FP16 AMP Optimized</span>
          </div>
        </div>
      </footer>

      {/* Official CWC / NDRF Incident Briefing Dossier Modal */}
      <IncidentBriefingModal
        isOpen={isBriefingModalOpen}
        onClose={() => setIsBriefingModalOpen(false)}
        basin={selectedBasin}
        horizon={selectedHorizon}
      />
    </main>
  );
}
