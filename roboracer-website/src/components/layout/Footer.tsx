const Footer = () => {
    return (
        <footer className="bg-brand-dark border-t border-white/10 pt-12 pb-8">
            <p className="text-center text-gray-500 text-sm">
                &copy; {new Date().getFullYear()} Penn RoboRacer Club. All rights reserved.
            </p>
        </footer>
    );
};

export default Footer;
