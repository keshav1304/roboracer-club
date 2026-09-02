import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import PageHero from "@/components/layout/PageHero";

interface DocumentationLayoutProps {
  title: string;
  subtitle: string;
  backHref?: string;
  backLabel?: string;
  children: React.ReactNode;
}

export default function DocumentationLayout({
  title,
  subtitle,
  backHref = "/roboracer-mini",
  backLabel = "Back to RoboRacer-mini",
  children,
}: DocumentationLayoutProps) {
  return (
    <div className="flex flex-col">
      <PageHero title={title} subtitle={subtitle} compact />

      <div className="bg-gradient-to-b from-brand-dark to-brand-[#ec4899]">
        <section className="py-14 px-4 sm:px-6 lg:px-8">
          <div className="max-w-3xl mx-auto">
            <Link
              href={backHref}
              className="inline-flex items-center gap-2 mb-10 text-gray-400 hover:text-white transition-colors text-sm font-medium"
            >
              <ArrowLeft size={16} />
              {backLabel}
            </Link>
            {children}
          </div>
        </section>
      </div>
    </div>
  );
}
