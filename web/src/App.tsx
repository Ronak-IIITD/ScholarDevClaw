import { Routes, Route, Navigate } from "react-router-dom";
import Layout from "@/components/Layout";
import DashboardPage from "@/pages/DashboardPage";
import SpecsPage from "@/pages/SpecsPage";
import PipelinePage from "@/pages/PipelinePage";
import PaperToCodePage from "@/pages/PaperToCodePage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/paper-to-code" element={<PaperToCodePage />} />
        <Route path="/specs" element={<SpecsPage />} />
        <Route path="/pipeline" element={<PipelinePage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
