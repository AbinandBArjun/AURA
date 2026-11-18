interface PlaceholderPageProps {
  title: string;
  description: string;
}

export default function PlaceholderPage({
  title,
  description,
}: PlaceholderPageProps) {
  return (
    <section className="placeholder-page">
      <p className="eyebrow">AURA WORKSPACE</p>
      <h1>{title}</h1>
      <p>{description}</p>

      <div className="placeholder-panel">
        <span>MODULE IN DEVELOPMENT</span>
        <p>
          This module will be connected to the AURA backend
          in the upcoming integration stages.
        </p>
      </div>
    </section>
  );
}
