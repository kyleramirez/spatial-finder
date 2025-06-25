
Features:
- upload OSM PBF to update data
- view / interact with vector map
  - base map
  - road map
- crud points of interest
- 
- follow self
- route to points of interest
  - estimate time en-route
- measure distance




I'm wanting to create an application with several layers for creating an offline map capable of running in Windows and accessing GPS via serial com 3. I plan to write the application partly using Rust and also using front-end typescript. I also need to develop a data pipeline to keep the app up to date on its own without having to go through a service or subscription. Here is how it'll work:

- app pre-downloads osm pbf file from geofabrik
- app whittles down data to a base map with roads and administrative boundaries based on what's available in the pbf
- app converts the data to GeoJSON, and eventually into PMTiles
- app serves a web app via tauri as a desktop application
- app hosts an endpoint for osrm routing engine to use openstreetmaps data to create a route in geojson with directions, and uses lua configuration to create weighted directions such as using longer travel times for dirt roads
- app listens on a serialport like:
  let port = serialport::new("COM3", 9600).open().expect("Failed to open port");
  so that a user can connect their XGPS150 to the windows laptop and the app listens to the data to provide real-time position updates within the app
- the front end is a small site using maplibre gl
- the front end allows the user to create a trip and either complete or exit the trip
- the app keeps itself up to date
- ideally this app provides a browser dev environment, and can at any time build an executable either for MacOS or an .exe to test the whole thing. I want to use vite and typescript and svelte, ideally, and the only reason I need a backend at all is to be able to serve the PMtiles via byte range requests, and also the endpoint to create a route using osrm, and hopefully a websocket endpoint to keep track of the user's current location through the GPS updates. The trips can live in browser memory and should only be one trip at a time per client. 

Based on this description, how should it be architected?

