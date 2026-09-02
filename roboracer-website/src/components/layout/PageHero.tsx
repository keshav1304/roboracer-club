import Image from "next/image";

interface PageHeroProps {
    title: string;
    subtitle: string;
    compact?: boolean;
}

export default function PageHero({ title, subtitle, compact = false }: PageHeroProps) {
    return (
        <section className={`relative overflow-hidden ${compact ? "py-12" : "py-16"}`}>
            <div className="absolute inset-0 z-0">
                <Image
                    src="/images/hero_background.png"
                    alt={title}
                    fill
                    className="object-cover opacity-40"
                    priority
                />
                <div className="absolute inset-0 bg-brand-dark/70" />
            </div>

            <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
                <h1 className={`text-5xl md:text-7xl font-bold leading-normal ${compact ? "mb-4" : "mb-6"}`}>
                    <span className="inline-block py-1 bg-clip-text text-transparent bg-gradient-to-r from-brand-blue to-brand-magenta">
                        {title}
                    </span>
                </h1>
                <p className="text-xl md:text-2xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
                    {subtitle}
                </p>
            </div>
        </section>
    );
}
