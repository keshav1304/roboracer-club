interface DocVideoProps {
  youtubeId: string;
  title?: string;
}

export default function DocVideo({ youtubeId, title }: DocVideoProps) {
  return (
    <figure className="my-8">
      <div className="relative aspect-video rounded-xl overflow-hidden border border-brand-magenta/30 shadow-lg shadow-brand-magenta/10">
        <iframe
          src={`https://www.youtube.com/embed/${youtubeId}`}
          title={title ?? "Documentation video"}
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
          allowFullScreen
          className="absolute inset-0 w-full h-full"
        />
      </div>
      {title && (
        <figcaption className="mt-3 text-sm text-gray-400 text-center italic">
          {title}
        </figcaption>
      )}
    </figure>
  );
}
