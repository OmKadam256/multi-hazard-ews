// ==============================================================================
// NATIONAL MULTI-HAZARD SATELLITE INTELLIGENCE OBSERVATORY (INDIA)
// Platform: Google Earth Engine (Project: multi-hazard-eews)
// Hazards: Flood Inundation (Sentinel-1 SAR) + Drought Vigor (MODIS NDVI + CHIRPS)
// ==============================================================================

// 1. Set View over Indian Subcontinent
Map.setCenter(79.5, 21.5, 5);
Map.setOptions('HYBRID');

// 2. CHIRPS v2.0 Cumulative Monsoon Rainfall (mm)
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate('2024-07-01', '2024-07-20')
  .sum();

var rainPalette = ['#f0f9ff', '#bae6fd', '#38bdf8', '#0284c7', '#1e3a8a', '#4c1d95'];
Map.addLayer(chirps, {min: 25, max: 350, palette: rainPalette}, '1. CHIRPS 20-Day Rainfall (mm)', true);

// 3. Copernicus 30m Global DEM (GLO-30) - Topography & Slope
var dem = ee.ImageCollection('COPERNICUS/DEM/GLO30')
  .select('DEM')
  .mosaic();

var slope = ee.Terrain.slope(dem);
var slopePalette = ['#22c55e', '#a3e635', '#facc15', '#fb923c', '#ef4444'];
Map.addLayer(slope, {min: 0, max: 25, palette: slopePalette}, '2. Terrain Slope / Orographic Relief (deg)', false);

// 4. Sentinel-1 C-SAR Dual Polarized Radar (Flood Extent)
var sar = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filterDate('2024-07-01', '2024-07-20')
  .select('VV')
  .median();

// Water surfaces have very low backscatter (< -16 dB)
var waterMask = sar.lt(-16.0).rename('flood_water');
Map.addLayer(sar, {min: -25, max: 0, palette: ['#020617', '#475569', '#cbd5e1']}, '3. Sentinel-1 Radar Backscatter (VV)', false);
Map.addLayer(waterMask.updateMask(waterMask), {palette: ['#00ffff']}, '4. Detected Surface Water & Inundation', false);

// 5. MODIS 16-Day NDVI (Vegetation Vigor & Agricultural Drought)
var modisNDVI = ee.ImageCollection('MODIS/061/MOD13A2')
  .filterDate('2024-06-01', '2024-07-20')
  .select('NDVI')
  .median()
  .multiply(0.0001); // Scale factor

var ndviPalette = ['#78350f', '#d97706', '#fef08a', '#84cc16', '#15803d'];
Map.addLayer(modisNDVI, {min: 0.1, max: 0.8, palette: ndviPalette}, '5. Agricultural Crop Vigor (NDVI)', false);

// 6. Major Indian Catchment Boundaries (HydroSHEDS Level-5)
var basins = ee.FeatureCollection('WWF/HydroSHEDS/v1/Basins/hybas_5')
  .filterBounds(ee.Geometry.Polygon([
    [[68.0, 6.0], [97.5, 6.0], [97.5, 36.5], [68.0, 36.5]]
  ]));

var basinOutline = ee.Image().paint({
  featureCollection: basins,
  color: 1,
  width: 1.5
});
Map.addLayer(basinOutline, {palette: ['#facc15']}, '6. HydroSHEDS River Basin Boundaries', true);

// 7. Interactive UI Legend & Summary in Console
print('====================================================');
print(' GeoAlert AI: National Earth Engine Intelligence Hub');
print(' Project: multi-hazard-eews (Active & Free)');
print('====================================================');
print('Active Sensor Layers Available:');
print(' - CHIRPS v2.0 (Rainfall Accumulation)');
print(' - Copernicus GLO-30 (Topography / Slope)');
print(' - Sentinel-1 C-SAR (Synthetic Aperture Radar Flood Water)');
print(' - MODIS Terra (NDVI Crop & Drought Stress)');
print(' - HydroSHEDS (Sub-continental River Basins)');
print('Use the "Layers" button in the top-right of the map to toggle any sensor!');
