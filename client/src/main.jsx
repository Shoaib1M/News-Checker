/*
FILE PURPOSE:
This is the main entry point for the React application. 
It takes our root React component (`App`) and attaches it to the real HTML document.

FLOW:
1. Imports React libraries.
2. Imports the global CSS (`index.css`).
3. Finds the `<div id="root"></div>` in the HTML file.
4. Renders the `<App />` component inside that div.

WHY THIS EXISTS:
React needs a way to bridge the gap between its virtual component tree and the actual browser DOM.
*/

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  // StrictMode helps find potential bugs by running checks and warnings in development mode.
  <StrictMode>
    <App />
  </StrictMode>,
)
