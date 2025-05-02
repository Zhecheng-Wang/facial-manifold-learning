import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';
import 'uikit/dist/css/uikit.min.css';
import UIkit from 'uikit';
import Icons from 'uikit/dist/js/uikit-icons';

// Load UIkit icons
UIkit.use(Icons);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
); 