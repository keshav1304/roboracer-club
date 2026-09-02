import React, { isValidElement, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import { getGithubAssetBase, headingId, type DocSlug } from "@/lib/docs";
import DocVideo from "./DocVideo";

function extractYoutubeId(src?: string): string | null {
  if (!src) return null;
  const match = src.match(/img\.youtube\.com\/vi\/([^/]+)/);
  return match?.[1] ?? null;
}

function getTextContent(children: ReactNode): string {
  if (typeof children === "string") return children;
  if (Array.isArray(children)) return children.map(getTextContent).join("");
  if (isValidElement<{ children?: ReactNode }>(children)) {
    return getTextContent(children.props.children);
  }
  return "";
}

interface DocumentationContentProps {
  content: string;
  assetBase: DocSlug;
}

export default function DocumentationContent({
  content,
  assetBase,
}: DocumentationContentProps) {
  return (
    <article className="doc-prose max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={{
          h1: ({ children }) => (
            <h1 className="text-3xl md:text-4xl font-bold mb-6 mt-2 leading-normal">
              <span className="inline-block py-0.5 bg-clip-text text-transparent bg-gradient-to-r from-brand-blue to-brand-magenta">
                {children}
              </span>
            </h1>
          ),
          h2: ({ children }) => {
            const text = getTextContent(children);
            const id = headingId(text);
            const isTldr = text === "TL;DR";
            return (
              <h2
                id={id}
                className={`text-2xl md:text-3xl font-bold mb-5 text-white scroll-mt-24 ${
                  isTldr ? "mt-0 pt-0 border-0" : "mt-14 pt-6 border-t border-white/10"
                }`}
              >
                {children}
              </h2>
            );
          },
          h3: ({ children }) => {
            const id = headingId(getTextContent(children));
            return (
              <h3 id={id} className="text-xl font-bold mt-10 mb-4 text-white scroll-mt-24">
                {children}
              </h3>
            );
          },
          div: ({ className, children }) => {
            if (className === "doc-callout") {
              return (
                <div className="my-8 p-6 md:p-8 rounded-2xl bg-brand-blue/5 border border-brand-blue/30 [&_p:last-child]:mb-0">
                  {children}
                </div>
              );
            }
            return <div className={className}>{children}</div>;
          },
          h4: ({ children }) => (
            <h4 className="text-lg font-semibold mt-8 mb-3 text-gray-200">{children}</h4>
          ),
          p: ({ children }) => (
            <p className="text-gray-300 leading-relaxed mb-5 text-lg">{children}</p>
          ),
          a: ({ href, children }) => {
            const childArray = React.Children.toArray(children);
            if (childArray.length === 1 && isValidElement<{ src?: string; alt?: string }>(childArray[0])) {
              const img = childArray[0];
              const youtubeId = extractYoutubeId(img.props.src);
              if (youtubeId) {
                return <DocVideo youtubeId={youtubeId} title={img.props.alt} />;
              }
            }
            if (href?.includes("youtube.com") || href?.includes("youtu.be")) {
              const id =
                href.match(/[?&]v=([^&]+)/)?.[1] ??
                href.match(/youtu\.be\/([^?]+)/)?.[1] ??
                href.match(/shorts\/([^?]+)/)?.[1];
              if (id) {
                return <DocVideo youtubeId={id} title={getTextContent(children)} />;
              }
            }
            return (
              <a
                href={href}
                target={href?.startsWith("http") ? "_blank" : undefined}
                rel={href?.startsWith("http") ? "noopener noreferrer" : undefined}
                className="text-brand-blue hover:text-brand-blue-hover underline underline-offset-2"
              >
                {children}
              </a>
            );
          },
          img: ({ src, alt }) => {
            const srcStr = typeof src === "string" ? src : undefined;
            const youtubeId = extractYoutubeId(srcStr);
            if (youtubeId) {
              return <DocVideo youtubeId={youtubeId} title={alt} />;
            }
            const resolved = srcStr?.startsWith("http")
              ? srcStr
              : `${getGithubAssetBase(assetBase)}/${srcStr?.replace(/^pictures\//, "") ?? ""}`;
            return (
              <figure className="my-8">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={resolved}
                  alt={alt ?? ""}
                  className="rounded-xl border border-white/10 max-w-full h-auto mx-auto"
                />
                {alt && (
                  <figcaption className="mt-3 text-sm text-gray-400 text-center italic">
                    {alt}
                  </figcaption>
                )}
              </figure>
            );
          },
          ul: ({ children }) => (
            <ul className="list-disc list-outside pl-6 mb-6 space-y-2 text-gray-300 text-lg">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal list-outside pl-6 mb-6 space-y-2 text-gray-300 text-lg">
              {children}
            </ol>
          ),
          li: ({ children }) => <li className="leading-relaxed">{children}</li>,
          strong: ({ children }) => (
            <strong className="font-semibold text-white">{children}</strong>
          ),
          em: ({ children }) => <em className="text-gray-200 italic">{children}</em>,
          hr: () => null,
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-brand-magenta/50 pl-5 my-6 text-gray-400 italic">
              {children}
            </blockquote>
          ),
          pre: ({ children }) => (
            <pre className="my-6 p-5 rounded-xl bg-brand-dark border border-white/10 overflow-x-auto text-sm leading-relaxed">
              {children}
            </pre>
          ),
          code: ({ className, children }) => {
            const isBlock = className?.includes("language-");
            if (isBlock) {
              return (
                <code className="font-mono text-brand-blue/90 text-sm">{children}</code>
              );
            }
            return (
              <code className="px-1.5 py-0.5 rounded bg-white/10 text-brand-blue font-mono text-sm">
                {children}
              </code>
            );
          },
          table: ({ children }) => (
            <div className="my-8 overflow-x-auto rounded-xl border border-white/10">
              <table className="w-full text-left text-sm">{children}</table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-white/5 text-gray-200 uppercase text-xs tracking-wider">
              {children}
            </thead>
          ),
          tbody: ({ children }) => <tbody className="divide-y divide-white/10">{children}</tbody>,
          tr: ({ children }) => <tr className="hover:bg-white/5 transition-colors">{children}</tr>,
          th: ({ children }) => (
            <th className="px-4 py-3 font-semibold border-b border-white/10">{children}</th>
          ),
          td: ({ children }) => <td className="px-4 py-3 text-gray-300">{children}</td>,
        }}
      >
        {content}
      </ReactMarkdown>
    </article>
  );
}
