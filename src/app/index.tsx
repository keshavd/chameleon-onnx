/**
 *
 * App
 *
 * This component is the skeleton around the actual pages, and should only
 * contain code that should be seen on all pages. (e.g. navigation bar)
 */

import * as React from 'react';
import { Helmet } from 'react-helmet-async';
import { Switch, Route, BrowserRouter } from 'react-router-dom';

import { GlobalStyle } from '../styles/global-styles';
import { useTranslation } from 'react-i18next';
import ChameleonOnnx from '../app/pages/onnx-demo'

export function App() {
  const { i18n } = useTranslation();
  const ort = require('onnxruntime-web');
  return (
    <BrowserRouter>
      <Helmet
        titleTemplate="%s - React Boilerplate"
        defaultTitle="React Boilerplate"
        htmlAttributes={{ lang: i18n.language }}
      >
        <meta name="description" content="A React Boilerplate application" />
      </Helmet>

      <Switch>
        <Route exact path={process.env.PUBLIC_URL + '/'} component={ChameleonOnnx} />
      </Switch>
      <GlobalStyle />
    </BrowserRouter>
  );
}
