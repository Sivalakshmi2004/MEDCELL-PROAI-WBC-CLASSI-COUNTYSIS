import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import MainPage from './MainPage'; // Your main page component
import DashboardPage from './DashboardPage'; // Your dashboard page component

const AppRouter = () => {
  return (
    <Router>
      <Switch>
        <Route path="/" exact component={MainPage} />
        <Route path="/dashboard" exact component={DashboardPage} />
      </Switch>
    </Router>
  );
};

export default AppRouter;
