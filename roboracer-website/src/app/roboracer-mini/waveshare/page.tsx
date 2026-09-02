import DocumentationContent from "@/components/documentation/DocumentationContent";
import DocumentationLayout from "@/components/documentation/DocumentationLayout";
import DocSourceFooter from "@/components/documentation/DocSourceFooter";
import DocTableOfContents from "@/components/documentation/DocTableOfContents";
import { readDoc } from "@/lib/docs";

export const revalidate = 60;

export const metadata = {
  title: "Waveshare PiRacer Pro Documentation | Penn RoboRacer Club",
  description:
    "Donkeycar on the Waveshare PiRacer Pro: hardware setup, PWM tuning, line following, lane keeping, and RoboRacer-mini evaluation.",
};

export default async function WaveshareDocumentationPage() {
  const { content, toc, sourceUrl } = await readDoc("waveshare");

  return (
    <DocumentationLayout
      title="Waveshare PiRacer Pro"
      subtitle="Investigating the Waveshare PiRacer Pro for RoboRacer-mini"
    >
      <DocTableOfContents headings={toc} />
      <DocumentationContent content={content} assetBase="waveshare" />
      <DocSourceFooter url={sourceUrl} />
    </DocumentationLayout>
  );
}
