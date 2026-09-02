import PageHero from "@/components/layout/PageHero";

export default function GalleryPage() {
    const images = Array.from({ length: 9 }).map((_, i) => ({
        id: i,
        alt: `Gallery Image ${i + 1}`,
        aspect: i % 2 === 0 ? "aspect-video" : "aspect-square"
    }));

    return (
        <div className="flex flex-col">
            <PageHero
                title="Gallery"
                subtitle="Highlights from events, races, and build sessions."
            />

            <div className="bg-gradient-to-b from-brand-dark to-brand-[#ec4899]">
                <section className="py-16 px-4 sm:px-6 lg:px-8">
                    <div className="max-w-5xl mx-auto">
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {images.map((img) => (
                                <div
                                    key={img.id}
                                    className={`bg-white/5 rounded-xl overflow-hidden border border-white/10 relative hover:scale-[1.02] transition-transform duration-300 ${img.aspect}`}
                                >
                                    <div className="absolute inset-0 flex items-center justify-center bg-gray-900 text-gray-700">
                                        <span className="text-lg font-medium">Image Placeholder {img.id + 1}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>
            </div>
        </div>
    );
}
