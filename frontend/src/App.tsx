import { Navigate, Route, Routes } from "react-router-dom";

import AppLayout from "./components/layout/AppLayout";
import Dashboard from "./pages/Dashboard";
import PlaceholderPage from "./pages/PlaceholderPage";

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route path="/" element={<Dashboard />} />

        <Route
          path="/documents"
          element={
            <PlaceholderPage
              title="Document Workspace"
              description="Manage and organize your research documents."
            />
          }
        />

        <Route
          path="/search"
          element={
            <PlaceholderPage
              title="Knowledge Search"
              description="Search your indexed knowledge base."
            />
          }
        />

        <Route
          path="/research"
          element={
            <PlaceholderPage
              title="Research Agents"
              description="Execute autonomous, evidence-grounded research."
            />
          }
        />

        <Route
          path="/reports"
          element={
            <PlaceholderPage
              title="Research Reports"
              description="Explore generated research reports and citations."
            />
          }
        />

        <Route
          path="/analytics"
          element={
            <PlaceholderPage
              title="Analytics"
              description="Monitor research and retrieval performance."
            />
          }
        />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
