import { ExternalLink } from "lucide-react";

interface DocSourceFooterProps {
  url: string;
}

export default function DocSourceFooter({ url }: DocSourceFooterProps) {
  return (
    <footer className="mt-16 pt-8 border-t border-white/10">
      <a
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center gap-2 text-sm text-gray-500 hover:text-gray-300 transition-colors"
      >
        View original markdown on GitHub
        <ExternalLink size={14} />
      </a>
    </footer>
  );
}
