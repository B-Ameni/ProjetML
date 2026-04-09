import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import Training from "./pages/Training";
import Results from "./pages/Results";
import Dashboard from "./pages/Dashboard";
import Prediction from "./pages/Prediction";
import HistoryPage from "./pages/History";

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/training" element={<Training />} />
          <Route path="/prediction" element={<Prediction />} />
          <Route path="/results" element={<Results />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/history" element={<HistoryPage />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;