"use client";

import Link from "next/link";
import { useState } from "react";
import { ChevronDown, ArrowRight, FileText, ImageIcon, Play, ExternalLink } from "lucide-react";

export interface InvestigationDetail {
    label: string;
    value: string;
}

export interface InvestigationVideo {
    title: string;
    youtubeId: string;
    url: string;
}

export interface InvestigationCardProps {
    name: string;
    summary: string;
    status: string;
    details: InvestigationDetail[];
    photoSlots?: number;
    videoSlots?: number;
    videos?: InvestigationVideo[];
    documentationHref?: string;
    documentationDescription?: string;
}

export default function InvestigationCard({
    name,
    summary,
    status,
    details,
    photoSlots = 0,
    videoSlots = 0,
    videos = [],
    documentationHref,
    documentationDescription,
}: InvestigationCardProps) {
    const [isOpen, setIsOpen] = useState(false);
    const hasPhotos = photoSlots > 0;
    const hasVideoPlaceholders = videoSlots > 0 && videos.length === 0;
    const hasMedia = hasPhotos || videos.length > 0 || hasVideoPlaceholders;

    return (
        <div className="rounded-3xl bg-white/5 border border-white/10 overflow-hidden transition-colors duration-300 hover:border-brand-blue/50">
            <button
                type="button"
                onClick={() => setIsOpen(!isOpen)}
                className="w-full p-8 text-left flex items-start justify-between gap-4"
                aria-expanded={isOpen}
            >
                <div className="flex-1">
                    <div className="flex flex-wrap items-center gap-3 mb-3">
                        <h3 className="text-2xl font-bold">{name}</h3>
                        <span className="text-xs uppercase tracking-widest px-3 py-1 rounded-full bg-brand-blue/10 text-brand-blue border border-brand-blue/30">
                            {status}
                        </span>
                        {documentationHref && (
                            <span className="text-xs uppercase tracking-widest px-3 py-1 rounded-full bg-brand-magenta/10 text-brand-magenta border border-brand-magenta/30">
                                Documentation
                            </span>
                        )}
                    </div>
                    <p className="text-gray-400 text-lg">{summary}</p>
                    {documentationHref && (
                        <Link
                            href={documentationHref}
                            onClick={(e) => e.stopPropagation()}
                            className="inline-flex items-center gap-2 mt-4 text-brand-magenta hover:text-brand-magenta-hover font-medium transition-colors"
                        >
                            <FileText size={16} />
                            Read full documentation
                            <ArrowRight size={14} />
                        </Link>
                    )}
                </div>
                <ChevronDown
                    className={`h-6 w-6 text-brand-blue shrink-0 mt-1 transition-transform duration-300 ${isOpen ? "rotate-180" : ""}`}
                />
            </button>

            <div
                className={`grid transition-all duration-300 ease-in-out ${isOpen ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"}`}
            >
                <div className="overflow-hidden">
                    <div className="px-8 pb-8 pt-0 border-t border-white/10">
                        {documentationHref && (
                            <div className="mt-6 p-6 rounded-2xl bg-brand-dark border-2 border-brand-magenta/40">
                                <div className="flex items-start gap-4">
                                    <div className="w-12 h-12 rounded-xl bg-brand-magenta/10 flex items-center justify-center shrink-0">
                                        <FileText className="h-6 w-6 text-brand-magenta" />
                                    </div>
                                    <div className="flex-1">
                                        <h4 className="text-lg font-bold mb-2">Investigation Documentation</h4>
                                        {documentationDescription && (
                                            <p className="text-gray-400 text-sm leading-relaxed mb-4">
                                                {documentationDescription}
                                            </p>
                                        )}
                                        <p className="text-gray-500 text-sm mb-4">
                                            Full write-up with videos, photos, setup notes, and
                                            RoboRacer-mini evaluation.
                                        </p>
                                        <Link
                                            href={documentationHref}
                                            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-brand-magenta hover:bg-brand-magenta-hover text-white font-semibold text-sm transition-colors"
                                        >
                                            Open documentation
                                            <ArrowRight size={14} />
                                        </Link>
                                    </div>
                                </div>
                            </div>
                        )}

                        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-6 mt-6">
                            {details.map((detail) => (
                                <div key={detail.label}>
                                    <dt className="text-sm uppercase tracking-wider text-gray-500 mb-1">
                                        {detail.label}
                                    </dt>
                                    <dd className="text-lg text-gray-200">{detail.value}</dd>
                                </div>
                            ))}
                        </dl>

                        {hasMedia && (
                            <div className="mt-10 pt-8 border-t border-white/10">
                                <h4 className="text-sm uppercase tracking-wider text-gray-500 mb-4">
                                    Photos &amp; Videos
                                </h4>

                                {hasPhotos && (
                                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-4">
                                        {Array.from({ length: photoSlots }).map((_, i) => (
                                            <div
                                                key={`photo-${i}`}
                                                className="aspect-video rounded-xl bg-white/5 border border-dashed border-white/20 flex flex-col items-center justify-center gap-2 text-gray-500"
                                            >
                                                <ImageIcon className="h-8 w-8 opacity-50" />
                                                <span className="text-xs">Photo {i + 1}</span>
                                            </div>
                                        ))}
                                    </div>
                                )}

                                {videos.length > 0 && (
                                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                        {videos.map((video) => (
                                            <div key={video.youtubeId}>
                                                <div className="relative aspect-video rounded-xl overflow-hidden border border-brand-magenta/30">
                                                    <iframe
                                                        src={`https://www.youtube.com/embed/${video.youtubeId}`}
                                                        title={video.title}
                                                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                                                        allowFullScreen
                                                        className="absolute inset-0 w-full h-full"
                                                    />
                                                </div>
                                                <a
                                                    href={video.url}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className="inline-flex items-center gap-1.5 mt-2 text-sm text-brand-magenta hover:text-brand-magenta-hover transition-colors"
                                                >
                                                    <Play size={12} />
                                                    {video.title}
                                                    <ExternalLink size={12} />
                                                </a>
                                            </div>
                                        ))}
                                    </div>
                                )}

                                {hasVideoPlaceholders && (
                                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                        {Array.from({ length: videoSlots }).map((_, i) => (
                                            <div
                                                key={`video-${i}`}
                                                className="aspect-video rounded-xl bg-white/5 border border-dashed border-brand-magenta/30 flex flex-col items-center justify-center gap-2 text-gray-500"
                                            >
                                                <Play className="h-10 w-10 text-brand-magenta/50" />
                                                <span className="text-xs">Video {i + 1}</span>
                                            </div>
                                        ))}
                                    </div>
                                )}

                                {(hasPhotos || hasVideoPlaceholders) && videos.length === 0 && (
                                    <p className="text-sm text-gray-500 mt-3">More media coming soon</p>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
