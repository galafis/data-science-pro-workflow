document.addEventListener('DOMContentLoaded', () => {
    console.log('GitHub Pages para data-science-pro-workflow carregado com sucesso!');

    // Exemplo de interatividade: rolagem suave para links de navegação
    document.querySelectorAll('nav ul li a').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href.startsWith('#')) {
                e.preventDefault();
                document.querySelector(href).scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
});
