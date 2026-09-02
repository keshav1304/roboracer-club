import DocumentationContent from "@/components/documentation/DocumentationContent";
import DocumentationLayout from "@/components/documentation/DocumentationLayout";
import DocSourceFooter from "@/components/documentation/DocSourceFooter";
import DocTableOfContents from "@/components/documentation/DocTableOfContents";
import { readDoc } from "@/lib/docs";

export const revalidate = 60;

export const metadata = {
  title: "Lehigh E116 Documentation | Penn RoboRacer Club",
  description:
    "Week-by-week E116 journal: Jetson and Traxxas hardware, ROS 2, RealSense, AprilTags, gap follow, and RoboRacer-mini evaluation.",
};

export default async function LehighDocumentationPage() {
  const { content, toc, sourceUrl } = await readDoc("lehigh");

  return (
    <DocumentationLayout
      title="Lehigh UniversityE116"
      subtitle="Journey through E116 course materials and platform"
    >
      <DocTableOfContents headings={toc} />
      <DocumentationContent content={content} assetBase="lehigh" />
      <DocSourceFooter url={sourceUrl} />
    </DocumentationLayout>
  );
}
